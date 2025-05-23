"""LLM for abduction

This module provides functionality for using LLMs to generate abductive hypotheses
based on SMT constraints for premises and conclusions. It includes tools for:
1. Formatting SMT constraints for LLM prompting
2. Generating abductive hypotheses using LLMs
3. Validating hypotheses using Z3
4. Evaluating LLM performance on abduction tasks
5. Feedback-based refinement using SMT counterexamples
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


@dataclass
class AbductionIterationResult:
    """Stores the result of a single iteration in feedback-based abduction."""
    hypothesis: Optional[z3.BoolRef] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    counterexample: Optional[Dict[str, Any]] = None
    llm_response: str = ""
    prompt: str = ""
    iteration: int = 0


@dataclass
class FeedbackAbductionResult(AbductionResult):
    """Stores the result of a feedback-based abduction attempt."""
    iterations: List[AbductionIterationResult] = field(default_factory=list)
    total_iterations: int = 0
    
    def add_iteration(self, iteration_result: AbductionIterationResult) -> None:
        """Add an iteration result to the history."""
        self.iterations.append(iteration_result)
        self.total_iterations = len(self.iterations)


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
    

class FeedbackLLMAbductor(LLMAbductor):
    """
    Uses LLMs to generate abductive hypotheses with SMT-based feedback.
    
    This abductor enhances the basic LLMAbductor by providing counterexample-driven
    feedback to the LLM, allowing it to refine its hypotheses through multiple iterations.
    """
    
    def __init__(self, llm: LLM, max_iterations: int = 5, temperature: float = 0.7):
        """
        Initialize the feedback-based LLM abductor.
        
        Args:
            llm: The LLM implementation to use
            max_iterations: Maximum number of feedback iterations
            temperature: Temperature for LLM generation
        """
        super().__init__(llm=llm, max_attempts=1, temperature=temperature)
        self.max_iterations = max_iterations
    
    def generate_counterexample(self, 
                               problem: AbductionProblem, 
                               hypothesis: z3.BoolRef) -> Optional[Dict[str, Any]]:
        """
        Generate a counterexample that shows why the hypothesis is invalid.
        
        Args:
            problem: The abduction problem
            hypothesis: The hypothesis to analyze
            
        Returns:
            Optional[Dict[str, Any]]: A counterexample assignment for variables, or None if no counterexample exists
        """
        # Create full premise with domain constraints
        premise = problem.premise
        domain = problem.domain_constraints
        full_premise = z3.And(domain, premise) if not z3.is_true(domain) else premise
        
        # Check consistency: premise ∧ hypothesis is satisfiable
        if not is_sat(z3.And(full_premise, hypothesis)):
            # Find a model that satisfies the premise but not the hypothesis
            s = z3.Solver()
            s.add(full_premise)
            s.add(z3.Not(hypothesis))
            if s.check() == z3.sat:
                model = s.model()
                return {str(v): model.eval(v, model_completion=True) for v in problem.variables}
            return None
        
        # Check sufficiency: premise ∧ hypothesis |= conclusion
        # If not sufficient, find a model that satisfies premise ∧ hypothesis but not conclusion
        if not is_entail(z3.And(full_premise, hypothesis), problem.conclusion):
            s = z3.Solver()
            s.add(full_premise)
            s.add(hypothesis)
            s.add(z3.Not(problem.conclusion))
            if s.check() == z3.sat:
                model = s.model()
                return {str(v): model.eval(v, model_completion=True) for v in problem.variables}
        
        return None
    
    def create_feedback_prompt(self, 
                              problem: AbductionProblem, 
                              previous_iterations: List[AbductionIterationResult],
                              last_counterexample: Dict[str, Any]) -> str:
        """
        Create a prompt with feedback from previous iterations.
        
        Args:
            problem: The abduction problem
            previous_iterations: Previous iteration results
            last_counterexample: The counterexample from the most recent attempt
            
        Returns:
            str: Prompt with feedback for the LLM
        """
        smt2_string = problem.to_smt2_string()
        
        # Extract information about the latest attempt
        last_iteration = previous_iterations[-1]
        last_hypothesis = last_iteration.hypothesis
        
        # Format the counterexample for display
        ce_formatted = "\n".join([f"{var} = {value}" for var, value in last_counterexample.items()])
        
        # Determine the specific issue with the hypothesis
        issue_description = ""
        if not last_iteration.is_consistent:
            issue_description = "Your hypothesis is inconsistent with the premise."
        elif not last_iteration.is_sufficient:
            issue_description = "Your hypothesis, combined with the premise, doesn't imply the conclusion."
        
        # Compile feedback from all previous iterations
        feedback_history = ""
        for i, iter_result in enumerate(previous_iterations):
            feedback_history += f"\nAttempt {i+1}:\n"
            feedback_history += f"Hypothesis: {iter_result.hypothesis}\n"
            feedback_history += f"Consistent: {iter_result.is_consistent}, Sufficient: {iter_result.is_sufficient}\n"
            if i < len(previous_iterations) - 1:  # Don't include the last counterexample twice
                if iter_result.counterexample:
                    ce_str = ", ".join([f"{var}={val}" for var, val in iter_result.counterexample.items()])
                    feedback_history += f"Counterexample: {ce_str}\n"
        
        prompt = f"""
You are an expert in logical abduction and SMT (Satisfiability Modulo Theories).

I have an abduction problem in SMT-LIB2 format:
```
{smt2_string}
```

The goal is to find an explanatory hypothesis ψ such that:
1. (premise ∧ ψ) is satisfiable (consistent)
2. (premise ∧ ψ) implies the conclusion: {problem.conclusion}

Your previous attempt was:
{last_hypothesis}

{issue_description}

I found a counterexample where your hypothesis does not work:
{ce_formatted}

This means there is a case where either:
- The premise and your hypothesis are not satisfiable together, or
- The premise and your hypothesis don't imply the conclusion

History of your previous attempts:
{feedback_history}

Please provide a revised hypothesis that addresses this counterexample.
Provide ONLY the SMT-LIB2 assertion for the revised hypothesis ψ as your complete answer.
DO NOT include any explanations, only provide the SMT-LIB2 assertion directly.
For example, your answer should look like:
(assert (formula))

or just:
(formula)
"""
        return prompt
    
    def abduce_with_feedback(self, problem: AbductionProblem) -> FeedbackAbductionResult:
        """
        Generate an abductive hypothesis using iterative feedback from counterexamples.
        
        Args:
            problem: The abduction problem
            
        Returns:
            FeedbackAbductionResult: The result of the feedback-based abduction
        """
        start_time = time.time()
        
        # Create initial prompt without feedback
        initial_prompt = self.create_prompt(problem)
        
        # Store all iteration results
        iteration_results = []
        
        # Initialize the overall result
        result = FeedbackAbductionResult(
            problem=problem,
            prompt=initial_prompt,
            execution_time=0
        )
        
        # Start with basic prompt for the first iteration
        current_prompt = initial_prompt
        
        for iteration in range(self.max_iterations):
            try:
                # Generate hypothesis from LLM
                llm_response = self.llm.generate(
                    prompt=current_prompt,
                    temperature=self.temperature + (iteration * 0.05),
                    max_tokens=1000
                )
                
                # Extract and parse SMT expression
                smt_string = extract_smt_from_llm_response(llm_response)
                if not smt_string:
                    print(f"No SMT expression found in LLM response on iteration {iteration+1}")
                    continue
                
                # Parse the hypothesis
                hypothesis = parse_smt2_string(smt_string, problem)
                if hypothesis is None:
                    print(f"Failed to parse SMT expression on iteration {iteration+1}")
                    continue
                
                # Validate the hypothesis
                is_consistent, is_sufficient = self.validate_hypothesis(problem, hypothesis)
                
                # Save iteration result regardless of validity
                iter_result = AbductionIterationResult(
                    hypothesis=hypothesis,
                    is_consistent=is_consistent,
                    is_sufficient=is_sufficient,
                    is_valid=is_consistent and is_sufficient,
                    llm_response=llm_response,
                    prompt=current_prompt,
                    iteration=iteration + 1
                )
                
                # If valid, we're done
                if is_consistent and is_sufficient:
                    iteration_results.append(iter_result)
                    break
                
                # Generate a counterexample for feedback
                counterexample = self.generate_counterexample(problem, hypothesis)
                iter_result.counterexample = counterexample
                iteration_results.append(iter_result)
                
                # If no counterexample could be generated, we can't provide feedback
                if counterexample is None:
                    print(f"Could not generate counterexample on iteration {iteration+1}")
                    break
                
                # Create a new prompt with feedback for the next iteration
                current_prompt = self.create_feedback_prompt(
                    problem=problem,
                    previous_iterations=iteration_results,
                    last_counterexample=counterexample
                )
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in abduce_with_feedback (iteration {iteration+1}): {error_msg}")
                result.error = error_msg
                break
        
        # Populate the final result
        for iter_result in iteration_results:
            result.add_iteration(iter_result)
        
        # Set the final hypothesis and validation status from the last iteration
        if iteration_results:
            last_iter = iteration_results[-1]
            result.hypothesis = last_iter.hypothesis
            result.is_consistent = last_iter.is_consistent
            result.is_sufficient = last_iter.is_sufficient
            result.is_valid = last_iter.is_valid
            result.llm_response = last_iter.llm_response
        
        result.execution_time = time.time() - start_time
        return result
    
    def abduce(self, problem: AbductionProblem) -> FeedbackAbductionResult:
        """
        Override the parent abduce method to use feedback-based abduction.
        
        Args:
            problem: The abduction problem
            
        Returns:
            FeedbackAbductionResult: The result of the feedback-based abduction
        """
        return self.abduce_with_feedback(problem)
    
