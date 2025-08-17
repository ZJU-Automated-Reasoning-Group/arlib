"""Prompt generation for LLM-based abduction."""

from typing import List, Dict, Any
from .data_structures import AbductionProblem, AbductionIterationResult


def create_basic_prompt(problem: AbductionProblem) -> str:
    """Build a basic instruction prompt for generating abductive hypotheses."""
    return f"""You are an expert in logical abduction and SMT.

Problem in SMT-LIB2 format:
```
{problem.to_smt2_string()}
```

Variables: {', '.join([str(var) for var in problem.variables])}

Find hypothesis ψ such that:
1. (premise ∧ ψ) is satisfiable
2. (premise ∧ ψ) implies: {problem.conclusion}

Provide ONLY the SMT-LIB2 formula for ψ. Examples:
(assert (formula))
or
(formula)

NO explanations, NO declare-const statements."""


def create_feedback_prompt(
    problem: AbductionProblem,
    previous_iterations: List[AbductionIterationResult],
    last_counterexample: Dict[str, Any]
) -> str:
    """Build a feedback-augmented prompt for iterative abduction."""
    last_iteration = previous_iterations[-1]
    ce_formatted = "\n".join([f"{var} = {value}" for var, value in last_counterexample.items()])

    issue = ("inconsistent with the premise" if not last_iteration.is_consistent
             else "doesn't imply the conclusion")

    history = ""
    for i, result in enumerate(previous_iterations[:-1]):
        ce_str = ", ".join([f"{var}={val}" for var, val in result.counterexample.items()]) if result.counterexample else ""
        history += f"Attempt {i+1}: {result.hypothesis} (Consistent: {result.is_consistent}, Sufficient: {result.is_sufficient})\n"
        if ce_str:
            history += f"Counterexample: {ce_str}\n"

    return f"""Problem:
```
{problem.to_smt2_string()}
```

Goal: Find ψ such that (premise ∧ ψ) is satisfiable and implies: {problem.conclusion}

Your previous attempt: {last_iteration.hypothesis}
Issue: Your hypothesis is {issue}.

Counterexample:
{ce_formatted}

Previous attempts:
{history}

Provide ONLY the revised SMT-LIB2 formula for ψ."""
