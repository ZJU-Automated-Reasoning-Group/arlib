"""LLM for abduction

This module provides functionality for using LLMs to generate abductive hypotheses
based on SMT constraints for premises and conclusions. It includes tools for:
1. Formatting SMT constraints for LLM prompting
2. Generating abductive hypotheses using LLMs
3. Validating hypotheses using Z3
4. Evaluating LLM performance on abduction tasks
"""

import time
import z3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from arlib.llm.abduct.base import LLM
#from arlib.abduction.abductor import check_abduct
from arlib.utils.z3_solver_utils import is_sat, is_entail
from arlib.llm.abduct.utils import extract_smt_from_llm_response, parse_smt2_string
from arlib.utils.z3_expr_utils import get_variables


@dataclass
class AbductionProblem:
    """Represents an abduction problem with SMT constraints."""
    premise: z3.BoolRef
    conclusion: z3.BoolRef
    description: str = ""
    variables: List[z3.ExprRef] = field(default_factory=list)
    domain_constraints: z3.BoolRef = None
    
    def __post_init__(self):
        """Extract variables if not provided."""
        if not self.variables:
            # Extract variables from premise and conclusion
            self.variables = list(set(get_variables(self.premise) + get_variables(self.conclusion)))
        
        # Set default domain constraints if not provided
        if self.domain_constraints is None:
            self.domain_constraints = z3.BoolVal(True)
    
    def to_smt2_string(self) -> str:
        """Convert the abduction problem to SMT-LIB2 format."""
        smt2 = []
        
        # Declare variables
        for var in self.variables:
            sort = var.sort()
            name = str(var)
            if z3.is_int(var):
                smt2.append(f"(declare-const {name} Int)")
            elif z3.is_real(var):
                smt2.append(f"(declare-const {name} Real)")
            elif z3.is_bool(var):
                smt2.append(f"(declare-const {name} Bool)")
            elif z3.is_bv(var):
                smt2.append(f"(declare-const {name} (_ BitVec {var.size()}))")
            else:
                smt2.append(f"(declare-const {name} {sort})")
        
        # Add domain constraints
        if str(self.domain_constraints) != "True":
            smt2.append(f"(assert {self.domain_constraints.sexpr()})")
        
        # Add premise
        smt2.append(f"(assert {self.premise.sexpr()})")
        
        # Add conclusion as a goal
        smt2.append(f";; Goal: find ψ such that:")
        smt2.append(f";; 1. (premise ∧ ψ) is satisfiable")
        smt2.append(f";; 2. (premise ∧ ψ) |= conclusion")
        smt2.append(f";; where conclusion is:")
        smt2.append(f";; {self.conclusion.sexpr()}")
        
        return "\n".join(smt2)
    
    def to_natural_language(self) -> str:
        """Generate a natural language description of the abduction problem."""
        nl_desc = []
        nl_desc.append("Abduction Problem:")
        
        if self.description:
            nl_desc.append(f"Description: {self.description}")
        
        var_types = {}
        for var in self.variables:
            if z3.is_int(var):
                var_types[str(var)] = "integer"
            elif z3.is_real(var):
                var_types[str(var)] = "real number"
            elif z3.is_bool(var):
                var_types[str(var)] = "boolean"
            else:
                var_types[str(var)] = str(var.sort())
        
        var_desc = ", ".join([f"{var} is a {typ}" for var, typ in var_types.items()])
        nl_desc.append(f"Variables: {var_desc}")
        
        if str(self.domain_constraints) != "True":
            nl_desc.append(f"Domain constraints: {self.domain_constraints}")
        
        nl_desc.append(f"Premise (Γ): {self.premise}")
        nl_desc.append(f"Conclusion (φ): {self.conclusion}")
        nl_desc.append("Goal: Find an explanation (ψ) such that:")
        nl_desc.append("1. Γ ∧ ψ is satisfiable (consistent)")
        nl_desc.append("2. Γ ∧ ψ |= φ (sufficient to imply the conclusion)")
        
        return "\n".join(nl_desc)


@dataclass
class AbductionResult:
    """Stores the result of an abduction attempt."""
    problem: AbductionProblem
    hypothesis: Optional[z3.BoolRef] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    llm_response: str = ""
    prompt: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        """Calculate validity based on consistency and sufficiency."""
        self.is_valid = self.is_consistent and self.is_sufficient



class LLMAbductor:
    """Uses LLMs to generate abductive hypotheses for given SMT constraints."""
    
    def __init__(self, llm: LLM, max_attempts: int = 3, temperature: float = 0.7):
        """
        Initialize the LLM abductor.
        
        Args:
            llm: The LLM implementation to use
            max_attempts: Maximum number of attempts to generate valid hypothesis
            temperature: Temperature for LLM generation
        """
        self.llm = llm
        self.max_attempts = max_attempts
        self.temperature = temperature
        
    def create_prompt(self, problem: AbductionProblem) -> str:
        """
        Create a prompt for the LLM based on the abduction problem.
        
        Args:
            problem: The abduction problem
            
        Returns:
            str: The prompt for the LLM
        """
        smt2_string = problem.to_smt2_string()
        
        # Create a more direct prompt that asks for just the SMT expression
        prompt = f"""
You are an expert in logical abduction and SMT (Satisfiability Modulo Theories).

I have an abduction problem in SMT-LIB2 format:
```
{smt2_string}
```

The variables used in this problem are:
{', '.join([str(var) for var in problem.variables])}

The goal is to find an explanatory hypothesis ψ such that:
1. (premise ∧ ψ) is satisfiable (consistent)
2. (premise ∧ ψ) implies the conclusion: {problem.conclusion}

Provide ONLY the SMT-LIB2 assertion for hypothesis ψ as your complete answer.
DO NOT include any explanations, only provide the SMT-LIB2 assertion directly.
For example, your answer should look like:
(assert (formula))

or just:
(formula)

DO NOT include declare-const statements or any other statements - ONLY the hypothesis formula.
"""
        return prompt
    
    def validate_hypothesis(self, 
                          problem: AbductionProblem, 
                          hypothesis: z3.BoolRef) -> Tuple[bool, bool]:
        """
        Validate if a hypothesis is consistent with the premise and sufficient to imply the conclusion.
        
        Args:
            problem: The abduction problem
            hypothesis: The generated hypothesis
            
        Returns:
            Tuple[bool, bool]: (is_consistent, is_sufficient)
        """
        # Skip validation for simple True hypotheses
        if z3.is_true(hypothesis):
            return True, False
        
        # Create full premise with domain constraints
        premise = problem.premise
        domain = problem.domain_constraints
        full_premise = z3.And(domain, premise) if not z3.is_true(domain) else premise
        
        # Check consistency: premise ∧ hypothesis is satisfiable
        is_consistent = is_sat(z3.And(full_premise, hypothesis))
        
        # If not consistent, no need to check sufficiency
        if not is_consistent:
            return False, False
        
        # Check sufficiency: premise ∧ hypothesis |= conclusion
        is_sufficient = is_entail(z3.And(full_premise, hypothesis), problem.conclusion)
        
        return is_consistent, is_sufficient
        
    def abduce(self, problem: AbductionProblem) -> AbductionResult:
        """
        Generate an abductive hypothesis for the given problem.
        
        Args:
            problem: The abduction problem
            
        Returns:
            AbductionResult: The result of the abduction attempt
        """
        start_time = time.time()
        prompt = self.create_prompt(problem)
        llm_response = ""
        
        for attempt in range(self.max_attempts):
            try:
                # Generate hypothesis from LLM
                llm_response = self.llm.generate(
                    prompt=prompt,
                    temperature=self.temperature + (attempt * 0.1),
                    max_tokens=1000
                )
                
                # Extract and parse SMT expression
                smt_string = extract_smt_from_llm_response(llm_response)
                if not smt_string:
                    print(f"No SMT expression found in LLM response on attempt {attempt+1}")
                    continue
                
                # Pass the problem for context in parsing
                hypothesis = parse_smt2_string(smt_string, problem)
                if hypothesis is None:
                    print(f"Failed to parse SMT expression on attempt {attempt+1}")
                    continue
                
                # Validate the hypothesis
                is_consistent, is_sufficient = self.validate_hypothesis(problem, hypothesis)
                
                result = AbductionResult(
                    problem=problem,
                    hypothesis=hypothesis,
                    is_consistent=is_consistent,
                    is_sufficient=is_sufficient,
                    llm_response=llm_response,
                    prompt=prompt,
                    execution_time=time.time() - start_time
                )
                
                # Return immediately if valid
                if is_consistent and is_sufficient:
                    return result
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in abduce: {error_msg}")
                return AbductionResult(
                    problem=problem,
                    error=error_msg,
                    llm_response=llm_response,
                    prompt=prompt,
                    execution_time=time.time() - start_time
                )
        
        # Return last result or error
        if 'result' in locals():
            return result
        else:
            return AbductionResult(
                problem=problem,
                error="Failed to generate a valid hypothesis after multiple attempts",
                llm_response=llm_response,
                prompt=prompt,
                execution_time=time.time() - start_time
            )
    
