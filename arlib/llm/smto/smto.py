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
import os
import tempfile
from typing import Dict, List, Optional, Any, Union

from arlib.llm.smto.oracles import OracleInfo, WhiteboxOracleInfo, OracleType
from arlib.llm.llmtool.LLM_utils import LLM
from arlib.llm.llmtool.logger import Logger
from arlib.llm.smto.whitebox import WhiteboxAnalyzer, ModelEvaluator
from arlib.llm.smto.utils import (
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
                 whitebox_analysis: bool = False,
                 temperature: float = 0.1,
                 system_role: str = "You are a experienced programmer and good at understanding programs written in mainstream programming languages."):
        """
        Initialize Orax solver
        
        Args:
            api_key: API key for LLM provider (if None, looks for env variable)
            model: LLM model to use
            provider: LLM provider to use ('openai' or 'anthropic')
            cache_dir: Directory to cache oracle results (None for no caching)
            explanation_level: Level of explanation detail ('none', 'basic', 'detailed')
            whitebox_analysis: Whether to perform whitebox analysis on components
            temperature: Temperature for LLM generation
            system_role: System role prompt for LLM
        """
        # Set up logger
        log_file = os.path.join(tempfile.gettempdir(), "orax_solver.log")
        self.logger = Logger(log_file)
        
        # Set up LLM directly
        self.llm = LLM(
            online_model_name=model,
            logger=self.logger,
            temperature=temperature,
            system_role=system_role
        )
        
        # Set up explanations
        self.explanation_logger = ExplanationLogger(level=explanation_level)
        
        # Set up Z3 solver
        self.solver = z3.Solver()
        
        # Set up cache
        self.cache = OracleCache(cache_dir=cache_dir)
        
        # Set up whitebox analyzer if enabled
        self.whitebox_analysis = whitebox_analysis
        if whitebox_analysis:
            self.whitebox_analyzer = WhiteboxAnalyzer(
                llm=self.llm,
                explanation_callback=self._add_explanation
            )
            self.model_evaluator = ModelEvaluator(
                llm=self.llm,
                explanation_callback=self._add_explanation
            )
        
        # Set up oracle registry
        self.oracles: Dict[str, OracleInfo] = {}
        self.whitebox_models: Dict[str, str] = {}  # Symbolic models for whitebox oracles

    def _add_explanation(self, message: str, level: str = "basic"):
        """Add explanation to the logger"""
        self.explanation_logger.log(message, level)

    def _add_explanation_detailed(self, data: Any, prefix: str = ""):
        """Add detailed explanation to the logger"""
        message = f"{prefix}{data}"
        self.explanation_logger.log(message, "detailed")

    def get_explanations(self) -> List[Dict[str, Any]]:
        """Get all logged explanations"""
        return self.explanation_logger.get_history()

    def register_oracle(self, oracle_info: OracleInfo):
        """Register an oracle with the solver"""
        self.oracles[oracle_info.name] = oracle_info
        
        # If whitebox analysis is enabled and this is a whitebox oracle, analyze it
        if (self.whitebox_analysis and 
            isinstance(oracle_info, WhiteboxOracleInfo)):
            symbolic_model = self.whitebox_analyzer.analyze_oracle(oracle_info)
            if symbolic_model:
                self.whitebox_models[oracle_info.name] = symbolic_model

    def add_constraint(self, constraint: z3.BoolRef):
        """Add a constraint to the solver"""
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
        """Validate model against all registered oracles"""
        for oracle_name, oracle_info in self.oracles.items():
            if not self._validate_oracle_calls(model, oracle_info):
                return False
        return True

    def _validate_oracle_calls(self, model: z3.ModelRef, oracle_info: OracleInfo) -> bool:
        """Validate all oracle calls in the model"""
        # Find all function applications for this oracle
        oracle_calls = self._find_oracle_calls(model, oracle_info.name)
        
        for call_inputs in oracle_calls:
            # Query the oracle
            oracle_result = self._query_oracle(oracle_info, call_inputs)
            
            if oracle_result is None:
                self._add_explanation(f"Oracle {oracle_info.name} failed to produce result for {call_inputs}")
                return False
            
            # Check if model's result matches oracle's result
            model_result = self._get_model_result(model, oracle_info.name, call_inputs)
            
            if not values_equal(oracle_result, model_result):
                self._add_explanation(f"Oracle mismatch for {oracle_info.name}{call_inputs}: "
                                   f"oracle={oracle_result}, model={model_result}")
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
            # Use LLM directly with temporary system role update
            original_system_role = self.llm.systemRole
            self.llm.systemRole = "You are a precise function evaluator."
            
            result, _, _ = self.llm.infer(prompt, is_measure_cost=False)
            
            # Restore original system role
            self.llm.systemRole = original_system_role
            
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

    def _find_oracle_calls(self, model: z3.ModelRef, oracle_name: str) -> List[Dict[str, Any]]:
        """Find all applications of an oracle function in the model."""
        calls = []
        for decl in model.decls():
            if decl.name() == oracle_name:
                # This is a function application, not a function declaration
                # We need to get the arguments and their values from the model
                args = [model.eval(arg, True) for arg in decl.children()]
                calls.append({f"arg{i}": z3_value_to_python(arg) for i, arg in enumerate(args)})
        return calls

    def _get_model_result(self, model: z3.ModelRef, oracle_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get the result of an oracle function application from the model."""
        for decl in model.decls():
            if decl.name() == oracle_name:
                # Check if this declaration matches the inputs
                args = [model.eval(arg, True) for arg in decl.children()]
                decl_inputs = {f"arg{i}": z3_value_to_python(arg) for i, arg in enumerate(args)}
                
                # If inputs match, return the model's result for this function application
                if decl_inputs == inputs:
                    return z3_value_to_python(model.eval(decl(), True))
        return None

    def clear_explanations(self):
        """Clear explanation history"""
        self.explanation_logger.clear()

    def get_symbolic_model(self, oracle_name: str) -> Optional[str]:
        """Get the symbolic model for a whitebox oracle"""
        oracle_info = self.oracles.get(oracle_name)
        if (oracle_info and 
            isinstance(oracle_info, WhiteboxOracleInfo) and 
            oracle_info.symbolic_model):
            return oracle_info.symbolic_model
        return None
