"""
Whitebox oracle analysis and model extraction functionality.
"""

import logging
import z3
from typing import Dict, Optional, Any, List

from arlib.llm.smto.oracles import WhiteboxOracleInfo, OracleAnalysisMode
from arlib.llm.llmtool.LLM_utils import LLM


class WhiteboxAnalyzer:
    """
    Analyzer for whitebox oracles that extracts symbolic models from
    documentation, source code, or other available information.
    """
    
    def __init__(self, llm: LLM, explanation_callback=None):
        """
        Initialize the whitebox analyzer.
        
        Args:
            llm: The LLM instance to use
            explanation_callback: Optional callback to log explanations
        """
        self.llm = llm
        self.explanation_callback = explanation_callback
    
    def analyze_oracle(self, oracle_info: WhiteboxOracleInfo) -> Optional[str]:
        """
        Analyze a whitebox oracle to derive a symbolic model.
        
        Args:
            oracle_info: The whitebox oracle information
            
        Returns:
            A symbolic model as a string, or None if analysis fails
        """
        if self.explanation_callback:
            self.explanation_callback(f"Analyzing whitebox oracle '{oracle_info.name}' "
                                     f"using {oracle_info.analysis_mode.value} mode")
        
        # Build analysis prompt based on the oracle information
        prompt = self._build_analysis_prompt(oracle_info)
        
        # Generate symbolic model using LLM
        try:
            # Use LLM directly with temporary system role update
            original_system_role = self.llm.systemRole
            self.llm.systemRole = "You are an expert in software analysis and formal methods."
            
            symbolic_model, _, _ = self.llm.infer(prompt, is_measure_cost=False)
            
            # Restore original system role
            self.llm.systemRole = original_system_role
            
            # Store the model in the oracle info
            oracle_info.symbolic_model = symbolic_model.strip()
            
            if self.explanation_callback:
                self.explanation_callback(f"Successfully derived symbolic model for '{oracle_info.name}'")
                
            return symbolic_model.strip()
        
        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"Failed to analyze whitebox oracle '{oracle_info.name}': {str(e)}")
            return None
    
    def _build_analysis_prompt(self, oracle_info: WhiteboxOracleInfo) -> str:
        """Build a prompt for analyzing a whitebox oracle"""
        prompt = f"""As an expert in software analysis, create a precise symbolic model for the following component:

Component name: {oracle_info.name}
Input types: {[str(t) for t in oracle_info.input_types]}
Output type: {str(oracle_info.output_type)}
Description: {oracle_info.description}

Examples:
"""
        for example in oracle_info.examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n"

        # Add content based on analysis mode
        if oracle_info.analysis_mode == OracleAnalysisMode.DOCUMENTATION and oracle_info.documentation:
            prompt += f"\nDocumentation:\n{oracle_info.documentation}\n"
        
        if oracle_info.analysis_mode == OracleAnalysisMode.SOURCE_CODE and oracle_info.source_code:
            prompt += f"\nSource code:\n{oracle_info.source_code}\n"
            
        if oracle_info.analysis_mode == OracleAnalysisMode.BINARY and oracle_info.binary_code:
            # We can't directly include binary, so we might include metadata or disassembly
            prompt += f"\nBinary code is available for analysis (metadata only).\n"
            
        if oracle_info.analysis_mode == OracleAnalysisMode.MIXED:
            # Include all available information
            if oracle_info.documentation:
                prompt += f"\nDocumentation:\n{oracle_info.documentation}\n"
            if oracle_info.source_code:
                prompt += f"\nSource code:\n{oracle_info.source_code}\n"
            if oracle_info.binary_code:
                prompt += f"\nBinary code is available for analysis (metadata only).\n"
        
        # Add external knowledge if available
        if oracle_info.external_knowledge:
            prompt += "\nAdditional relevant knowledge:\n"
            for item in oracle_info.external_knowledge:
                prompt += f"- {item}\n"
        
        prompt += """
Based on this information, please:

1. Identify the key behaviors and patterns in this component
2. Create a formal symbolic model that precisely captures the input-output relationship
3. Express this model using SMT expressions compatible with Z3
4. Ensure the model correctly handles all provided examples

Your model should be as concise as possible while capturing all essential behaviors. If you're unsure about certain behaviors, make reasonable assumptions based on the provided information.

Output your model directly in a format that could be directly evaluated with Z3, without additional explanations:
"""
        return prompt


class ModelEvaluator:
    """
    Evaluator for symbolic models extracted from whitebox oracles.
    """
    
    def __init__(self, llm: LLM, explanation_callback=None):
        """
        Initialize the model evaluator.
        
        Args:
            llm: The LLM instance to use
            explanation_callback: Optional callback to log explanations
        """
        self.llm = llm
        self.explanation_callback = explanation_callback
    
    def evaluate_model(self, 
                       oracle_info: WhiteboxOracleInfo, 
                       inputs: Dict[str, Any]) -> Optional[Any]:
        """
        Evaluate a symbolic model with the given inputs.
        
        Args:
            oracle_info: The whitebox oracle information
            inputs: The input values
            
        Returns:
            The evaluation result, or None if evaluation fails
        """
        if not oracle_info.symbolic_model:
            return None
        
        try:
            # Try to evaluate the model directly using Z3
            result = self._evaluate_with_z3(oracle_info, inputs)
            if result is not None:
                return result
                
            # Fall back to LLM-based evaluation
            if self.explanation_callback:
                self.explanation_callback("Direct Z3 evaluation failed, falling back to LLM evaluation")
                
            return self._evaluate_with_llm(oracle_info, inputs)
            
        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"Model evaluation failed: {str(e)}")
            return None

    def _evaluate_with_z3(self, 
                          oracle_info: WhiteboxOracleInfo, 
                          inputs: Dict[str, Any]) -> Optional[Any]:
        """
        Evaluate the symbolic model using Z3.
        
        Args:
            oracle_info: The whitebox oracle information
            inputs: The input values
            
        Returns:
            The evaluation result, or None if Z3 evaluation fails
        """
        try:
            # Create a Z3 solver
            solver = z3.Solver()
            
            # Create variables for the inputs
            z3_vars = {}
            for i, input_type in enumerate(oracle_info.input_types):
                var_name = f"arg{i}"
                if input_type == z3.IntSort():
                    z3_vars[var_name] = z3.Int(var_name)
                elif input_type == z3.RealSort():
                    z3_vars[var_name] = z3.Real(var_name)
                elif input_type == z3.BoolSort():
                    z3_vars[var_name] = z3.Bool(var_name)
                elif input_type == z3.StringSort():
                    z3_vars[var_name] = z3.String(var_name)
                else:
                    # Unsupported type for direct Z3 evaluation
                    return None
            
            # Add constraints for the input values
            for var_name, value in inputs.items():
                if var_name in z3_vars:
                    if isinstance(value, bool):
                        solver.add(z3_vars[var_name] == z3.BoolVal(value))
                    elif isinstance(value, int):
                        solver.add(z3_vars[var_name] == z3.IntVal(value))
                    elif isinstance(value, float):
                        solver.add(z3_vars[var_name] == z3.RealVal(value))
                    elif isinstance(value, str):
                        solver.add(z3_vars[var_name] == z3.StringVal(value))
            
            # Try to parse and evaluate the symbolic model
            # This is a simplified approach - in practice, you'd need more sophisticated parsing
            model_str = oracle_info.symbolic_model
            
            # Create output variable
            if oracle_info.output_type == z3.IntSort():
                output_var = z3.Int("output")
            elif oracle_info.output_type == z3.RealSort():
                output_var = z3.Real("output")
            elif oracle_info.output_type == z3.BoolSort():
                output_var = z3.Bool("output")
            elif oracle_info.output_type == z3.StringSort():
                output_var = z3.String("output")
            else:
                return None
            
            # This is a placeholder - in practice, you'd need to parse the model_str
            # and create appropriate Z3 constraints
            # For now, we'll just return None to fall back to LLM evaluation
            return None
            
        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"Z3 evaluation failed: {str(e)}")
            return None
    
    def _evaluate_with_llm(self, 
                          oracle_info: WhiteboxOracleInfo, 
                          inputs: Dict[str, Any]) -> Optional[Any]:
        """
        Evaluate the symbolic model using LLM.
        
        Args:
            oracle_info: The whitebox oracle information
            inputs: The input values
            
        Returns:
            The evaluation result, or None if LLM evaluation fails
        """
        prompt = f"""Based on the following symbolic model for function '{oracle_info.name}':

{oracle_info.symbolic_model}

Evaluate this model with these input values:
{inputs}

Return only the resulting output value, with no additional text.
"""
        try:
            # Use LLM directly with temporary system role update
            original_system_role = self.llm.systemRole
            self.llm.systemRole = "You are a precise formula evaluator."
            
            result, _, _ = self.llm.infer(prompt, is_measure_cost=False)
            
            # Restore original system role
            self.llm.systemRole = original_system_role
            
            # Parse the result based on output type
            result = result.strip()
            
            if oracle_info.output_type == z3.BoolSort():
                return result.lower() in ['true', '1', 'yes']
            elif oracle_info.output_type == z3.IntSort():
                return int(result)
            elif oracle_info.output_type == z3.RealSort():
                return float(result)
            elif oracle_info.output_type == z3.StringSort():
                # Remove quotes if present
                if result.startswith('"') and result.endswith('"'):
                    return result[1:-1]
                return result
            else:
                return result
        
        except Exception as e:
            if self.explanation_callback:
                self.explanation_callback(f"LLM evaluation failed: {str(e)}")
            return None 