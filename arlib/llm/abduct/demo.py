#!/usr/bin/env python
"""
Demo for LLM-based abduction using ZhipuAI.

This demonstrates how to use the LLMAbductor with ZhipuAI to solve
abductive reasoning problems. The demo creates several example problems
and shows how ZhipuAI can generate explanatory hypotheses.
"""

import os
import z3
import datetime
from arlib.llm.abduct import (
    AbductionProblem, 
    LLMAbductor, 
    AbductionEvaluator,
    EnvLoader
)
from arlib.llm.abduct.zhipu import ZhipuLLM

def create_example_problems():
    """Create a set of example abduction problems."""
    problems = []
    
    # Problem 0: Very simple arithmetic (x > 3)
    x = z3.Int('x')
    premise = x > 0
    conclusion = x > 2
    problems.append(AbductionProblem(
        premise=premise,
        conclusion=conclusion,
        description="Very simple arithmetic inequality"
    ))
    
    # Problem 1: Linear arithmetic
    x, y = z3.Ints('x y')
    premise = z3.And(x > 0, y > 0)
    conclusion = x + y > 5
    problems.append(AbductionProblem(
        premise=premise,
        conclusion=conclusion,
        description="Simple linear arithmetic problem"
    ))
    
    # Problem 2: Boolean logic
    a, b, c = z3.Bools('a b c')
    premise = z3.And(a == z3.Not(b), c == z3.Or(a, b))
    conclusion = c
    problems.append(AbductionProblem(
        premise=premise,
        conclusion=conclusion,
        description="Boolean logic problem"
    ))
    
    return problems

def run_single_example(llm_abductor, problem):
    """Run abduction on a single problem and print results."""
    print(f"Problem: {problem.description}")
    print(f"Premise: {problem.premise}")
    print(f"Conclusion: {problem.conclusion}")
    print(f"Variables: {', '.join([str(var) for var in problem.variables])}")
    
    # Run abduction
    start_time = datetime.datetime.now()
    result = llm_abductor.abduce(problem)
    end_time = datetime.datetime.now()
    
    # Print results
    print("\nResults:")
    print(f"Execution time: {end_time - start_time}")
    print(f"Generated hypothesis: {result.hypothesis}")
    print(f"Is consistent: {result.is_consistent}")
    print(f"Is sufficient: {result.is_sufficient}")
    print(f"Is valid: {result.is_valid}")
    
    if not result.is_valid:
        print("Reason for invalidity:")
        if not result.is_consistent:
            print("- The hypothesis is inconsistent with the premise")
        if not result.is_sufficient:
            print("- The hypothesis is not sufficient to entail the conclusion")
    
    print("\nLLM Response (raw):")
    print("=" * 80)
    print(result.llm_response)
    print("=" * 80)
    print("\n" + "-" * 80 + "\n")

def main():
    """Main demo function."""
    # Check if ZhipuAI API key is set using EnvLoader
    api_key = EnvLoader.get_env("ZHIPU_API_KEY")
    if not api_key:
        print("Error: ZHIPU_API_KEY not found in environment or .env file.")
        print("Please set it with: export ZHIPU_API_KEY=your_api_key")
        print("Or create a .env file in the project root with: ZHIPU_API_KEY=your_api_key")
        return
    
    # Create LLM and abductor
    print("Initializing ZhipuAI LLM...")
    llm = ZhipuLLM(model_name="glm-4-flash")  # Can also use other models like glm-4
    llm_abductor = LLMAbductor(llm, max_attempts=1, temperature=0.9)  # Higher temperature for more creative responses
    
    # Create example problems
    problems = create_example_problems()
    
    # Run examples (first 3 only)
    for i, problem in enumerate(problems[:3]):
        print(f"\nExample {i+1}/{len(problems[:3])}")
        run_single_example(llm_abductor, problem)

if __name__ == "__main__":
    main()
