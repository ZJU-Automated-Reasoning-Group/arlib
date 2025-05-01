"""
Orax: An SMTO solver using LLM as oracle handler

SMT solving has driven progress in formula-based software verification, but struggles
with open programs containing components lacking formal specifications (third-party libraries,
deep learning models). The Satisfiability Modulo Theories and Oracles (SMTO) problem
addresses this by treating black-box components as oracles with observable inputs/outputs
but unknown implementations.

This is the main SMTO solver implementation file, which coordinates the integration of
Z3 SMT solving with LLM-powered oracles in both blackbox and whitebox modes.
"""

import z3
from typing import Dict, List, Optional, Any, Union

from arlib.llm.oracles import OracleInfo, WhiteboxOracleInfo, OracleType
from arlib.llm.llm_providers import LLMInterface, LLMConfig
from arlib.llm.whitebox import WhiteboxAnalyzer, ModelEvaluator
from arlib.llm.utils import (
    OracleCache,
    ExplanationLogger,
    z3_value_to_python,
    python_to_z3_value,
    values_equal,
    generate_cache_key
)


class OraxSolver:
    """
    SMTO solver that combines Z3 SMT solving with LLM-powered oracles.
    
    Supports both blackbox and whitebox modes:
    - Blackbox: Traditional SMTO where we can only observe input-output behavior
    - Whitebox: Enhanced SMTO using LLM to analyze available component information
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-4", 
                 provider: str = "openai",
                 cache_dir: Optional[str] = None,
                 explanation_level: str = "basic",
                 whitebox_analysis: bool = False):
        """
        Initialize Orax solver
        
        Args:
            api_key: API key for LLM provider (if None, looks for env variable)
            model: LLM model to use
            provider: LLM provider to use ('openai' or 'anthropic')
            cache_dir: Directory to cache oracle results (None for no caching)
            explanation_level: Level of explanation detail ('none', 'basic', 'detailed')
            whitebox_analysis: Whether to perform whitebox analysis on components
        """
        # Set up LLM interface
        llm_config = LLMConfig(api_key=api_key, model=model, provider=provider)
        self.llm = LLMInterface(llm_config)
        
        # Set up explanations
        self.logger = ExplanationLogger(level=explanation_level)
        
        # Set up Z3 solver
        self.solver = z3.Solver()
        
        # Set up cache
        self.cache = OracleCache(cache_dir=cache_dir)
        
        # Set up whitebox analyzer if enabled
        self.whitebox_analysis = whitebox_analysis
        if whitebox_analysis:
            self.whitebox_analyzer = WhiteboxAnalyzer(
                llm_interface=self.llm,
                explanation_callback=self._add_explanation
            )
            self.model_evaluator = ModelEvaluator(
                llm_interface=self.llm,
                explanation_callback=self._add_explanation
            )
        
        # Set up oracle registry
        self.oracles: Dict[str, OracleInfo] = {}
        self.whitebox_models: Dict[str, str] = {}  # Symbolic models for whitebox oracles

    def register_oracle(self, oracle_info: OracleInfo):
        """Register an oracle function"""
        self.oracles[oracle_info.name] = oracle_info
        
        # Create Z3 function declaration if it doesn't exist
        try:
            # Check if function already exists
            z3.Function(oracle_info.name, *oracle_info.input_types, oracle_info.output_type)
        except z3.Z3Exception:
            # Create new function
            z3.FuncDecl(oracle_info.name, oracle_info.input_types, oracle_info.output_type)
            
        # If this is a whitebox oracle and whitebox analysis is enabled, analyze it
        if (isinstance(oracle_info, WhiteboxOracleInfo) and 
            self.whitebox_analysis and 
            oracle_info.oracle_type == OracleType.WHITEBOX):
            self._analyze_whitebox_oracle(oracle_info)

    def _analyze_whitebox_oracle(self, oracle_info: WhiteboxOracleInfo) -> None:
        """Analyze a whitebox oracle to derive a more accurate model"""
        if self.whitebox_analysis:
            symbolic_model = self.whitebox_analyzer.analyze_oracle(oracle_info)
            if symbolic_model:
                self.whitebox_models[oracle_info.name] = symbolic_model

    def add_constraint(self, constraint: z3.BoolRef):
        """Add constraint to the solver"""
        self.solver.add(constraint)

    def check(self, timeout_ms: int = 0) -> Optional[z3.ModelRef]:
        """
        Check satisfiability with oracle feedback loop
        
        Args:
            timeout_ms: Timeout in milliseconds (0 for no timeout)
            
        Returns model if satisfiable, None if unsatisfiable
        """
        if timeout_ms > 0:
            self.solver.set("timeout", timeout_ms)
            
        max_iterations = 10
        for iteration in range(max_iterations):
            # Log iteration start
            self._add_explanation(f"Starting SMTO iteration {iteration+1}/{max_iterations}")
            
            result = self.solver.check()
            if result == z3.unsat:
                self._add_explanation("Problem is unsatisfiable")
                return None
            
            model = self.solver.model()
            
            # Log model information
            self._add_explanation_detailed({str(decl()): str(model[decl]) for decl in model.decls()},
                                      prefix="Candidate model found: ")
            
            is_valid = self._validate_model_with_oracles(model)

            if is_valid:
                self._add_explanation("Valid model found satisfying all oracle constraints")
                return model

            # Add learned constraints from oracle feedback
            self._add_oracle_constraints(model)
            
            self._add_explanation(f"Model validation failed, adding oracle constraints and trying again")

        self._add_explanation("Maximum iterations reached without finding valid model")
        return None

    def _validate_model_with_oracles(self, model: z3.ModelRef) -> bool:
        """Validate model by checking oracle constraints"""
        for oracle_name, oracle_info in self.oracles.items():
            # Find all applications of this oracle in the model
            for decl in model.decls():
                if decl.name() == oracle_name:
                    args = [model.eval(arg, True) for arg in decl.children()]
                    expected = model.eval(decl(), True)
                    
                    # Process inputs based on oracle type
                    inputs = {f"arg{i}": z3_value_to_python(arg) for i, arg in enumerate(args)}
                    
                    # Query oracle for actual result
                    actual = self._query_oracle(oracle_info, inputs)
                    
                    if actual is None or not values_equal(actual, expected):
                        self._add_explanation_detailed(
                            f"Oracle validation failed for {oracle_name}: "
                            f"expected {expected}, got {actual}"
                        )
                        return False
        return True

    def _query_oracle(self, oracle_info: OracleInfo, inputs: Dict[str, Any]) -> Optional[Any]:
        """Query oracle based on type"""
        # Generate cache key
        cache_key = generate_cache_key(oracle_info.name, inputs)
        
        # Check cache first
        if self.cache.contains(cache_key):
            self._add_explanation_detailed(f"Using cached result for {oracle_info.name}{inputs}")
            return self.cache.get(cache_key)
        
        # Query based on oracle type
        result = None
        
        if oracle_info.oracle_type == OracleType.FUNCTION and oracle_info.function is not None:
            result = oracle_info.function(**inputs)
        elif oracle_info.oracle_type == OracleType.LLM:
            result = self._query_llm_oracle(oracle_info, inputs)
        elif oracle_info.oracle_type == OracleType.WHITEBOX:
            # For whitebox oracles, use the whitebox query method if analysis was successful
            if (self.whitebox_analysis and 
                isinstance(oracle_info, WhiteboxOracleInfo) and
                oracle_info.symbolic_model):
                result = self.model_evaluator.evaluate_model(oracle_info, inputs)
                # Fall back to standard LLM query if model evaluation fails
                if result is None:
                    result = self._query_llm_oracle(oracle_info, inputs)
            else:
                # Fall back to standard LLM query
                result = self._query_llm_oracle(oracle_info, inputs)
        elif oracle_info.oracle_type == OracleType.EXTERNAL:
            raise NotImplementedError("External oracle type not yet implemented")
        else:
            raise ValueError(f"Unsupported oracle type: {oracle_info.oracle_type}")
        
        # Cache result
        if result is not None:
            self.cache.put(cache_key, result)
        
        return result
        
    def _query_llm_oracle(self, oracle_info: OracleInfo, inputs: Dict[str, Any]) -> Optional[Any]:
        """Query LLM to simulate oracle function"""
        # Construct prompt with oracle description and examples
        prompt = f"Act as the following function:\n{oracle_info.description}\n\nExamples:\n"
        for example in oracle_info.examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n"

        prompt += f"\nNow, given the input: {inputs}\nWhat is the output?"

        try:
            result = self.llm.generate_text(
                prompt=prompt,
                system_prompt="You are a precise function evaluator."
            )
            
            # Parse the result based on output type
            return self._parse_llm_response(result, oracle_info.output_type)
        except Exception as e:
            self._add_explanation(f"LLM query failed: {e}")
            return None

    def _parse_llm_response(self, response: str, output_type: z3.SortRef) -> Optional[Any]:
        """Parse LLM response according to expected output type"""
        try:
            response = response.strip()
            if output_type == z3.BoolSort():
                return response.lower() in ['true', '1', 'yes']
            elif output_type == z3.IntSort():
                return int(response)
            elif output_type == z3.RealSort():
                return float(response)
            elif output_type == z3.StringSort():
                # Remove quotes if present
                if response.startswith('"') and response.endswith('"'):
                    return response[1:-1]
                return response
            else:
                return response
        except Exception as e:
            self._add_explanation(f"Error parsing LLM response: {e}")
            return None

    def _add_oracle_constraints(self, model: z3.ModelRef):
        """Add learned constraints from oracle feedback"""
        constraints_added = 0
        for oracle_name, oracle_info in self.oracles.items():
            # Get all applications of this oracle in the model
            applications = []
            for decl in model.decls():
                if decl.name() == oracle_name:
                    applications.append(decl)
            
            for decl in applications:
                args = [model.eval(arg, True) for arg in decl.children()]
                inputs = {f"arg{i}": z3_value_to_python(arg) for i, arg in enumerate(args)}
                
                # Generate cache key
                cache_key = generate_cache_key(oracle_name, inputs)
                
                # Get oracle result (from cache or by querying)
                if not self.cache.contains(cache_key):
                    result = self._query_oracle(oracle_info, inputs)
                else:
                    result = self.cache.get(cache_key)
                
                if result is not None:
                    # Convert Python value to Z3 value
                    z3_actual = python_to_z3_value(result, oracle_info.output_type)
                    
                    # Add constraint that this oracle application must equal actual result
                    constraint = decl() == z3_actual
                    self.solver.add(constraint)
                    constraints_added += 1
        
        if constraints_added > 0:
            self._add_explanation(f"Added {constraints_added} oracle constraints")

    def _add_explanation(self, message: str):
        """Add basic explanation"""
        self.logger.log(message, level="basic")

    def _add_explanation_detailed(self, message: Union[str, Dict], prefix: str = ""):
        """Add detailed explanation"""
        if isinstance(message, dict):
            message = f"{prefix}{message}"
        self.logger.log(message, level="detailed")

    def get_explanations(self) -> List[Dict[str, Any]]:
        """Get explanation history"""
        return self.logger.get_history()

    def clear_explanations(self):
        """Clear explanation history"""
        self.logger.clear()

    def get_symbolic_model(self, oracle_name: str) -> Optional[str]:
        """Get the symbolic model for a whitebox oracle"""
        oracle_info = self.oracles.get(oracle_name)
        if (oracle_info and 
            isinstance(oracle_info, WhiteboxOracleInfo) and 
            oracle_info.symbolic_model):
            return oracle_info.symbolic_model
        return None
