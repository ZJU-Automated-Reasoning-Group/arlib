"""
Whitebox oracle analysis and model extraction functionality.
"""

import logging
import z3
from typing import Dict, Optional, Any, List

from arlib.llm.oracles import WhiteboxOracleInfo, OracleAnalysisMode
from arlib.llm.llm_providers import LLMInterface


class WhiteboxAnalyzer:
    """
    Analyzer for whitebox oracles that extracts symbolic models from
    documentation, source code, or other available information.
    """
    
    def __init__(self, llm_interface: LLMInterface, explanation_callback=None):
        """
        Initialize the whitebox analyzer.
        
        Args:
            llm_interface: Interface to the LLM provider
            explanation_callback: Optional callback to log explanations
        """
        self.llm = llm_interface
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
            symbolic_model = self.llm.generate_text(
                prompt=prompt,
                system_prompt="You are an expert in software analysis and formal methods."
            )
            
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
    
    def __init__(self, llm_interface: LLMInterface, explanation_callback=None):
        """
        Initialize the model evaluator.
        
        Args:
            llm_interface: Interface to the LLM provider
            explanation_callback: Optional callback to log explanations
        """
        self.llm = llm_interface
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
        Evaluate the symbolic model using Z3 directly.
        
        Args:
            oracle_info: The whitebox oracle information
            inputs: The input values
            
        Returns:
            The evaluation result, or None if Z3 evaluation fails
        """
        try:
            # Create a solver
            solver = z3.Solver()
            
            # Create input variables and add constraints
            input_vars = {}
            for i, input_type in enumerate(oracle_info.input_types):
                input_name = f"input_{i}"
                if input_type == z3.IntSort():
                    input_vars[input_name] = z3.Int(input_name)
                    solver.add(input_vars[input_name] == inputs[f"arg{i}"])
                elif input_type == z3.RealSort():
                    input_vars[input_name] = z3.Real(input_name)
                    solver.add(input_vars[input_name] == inputs[f"arg{i}"])
                elif input_type == z3.BoolSort():
                    input_vars[input_name] = z3.Bool(input_name)
                    solver.add(input_vars[input_name] == inputs[f"arg{i}"])
                elif input_type == z3.StringSort():
                    input_vars[input_name] = z3.String(input_name)
                    solver.add(input_vars[input_name] == z3.StringVal(inputs[f"arg{i}"]))
                else:
                    # Unsupported type
                    return None
            
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
            
            # Prepare the model
            model_str = oracle_info.symbolic_model
            for i in range(len(oracle_info.input_types)):
                model_str = model_str.replace(f"arg{i}", f"input_{i}")
            
            # Evaluate the model
            model_expr = eval(model_str)
            solver.add(output_var == model_expr)
            
            # Check satisfiability
            if solver.check() == z3.sat:
                model = solver.model()
                
                # Extract output value
                output_val = model[output_var]
                
                # Convert output value to Python type
                if oracle_info.output_type == z3.IntSort():
                    return output_val.as_long()
                elif oracle_info.output_type == z3.RealSort():
                    return float(output_val.as_fraction())
                elif oracle_info.output_type == z3.BoolSort():
                    return z3.is_true(output_val)
                elif oracle_info.output_type == z3.StringSort():
                    return output_val.as_string()
                
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
            result = self.llm.generate_text(
                prompt=prompt, 
                system_prompt="You are a precise formula evaluator."
            )
            
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