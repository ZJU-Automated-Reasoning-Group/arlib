#!/usr/bin/env python3
"""
Demo script showing different prompt types for LLM-based Craig interpolant generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from arlib.llm.interpolant.llm_interpolant import LLMInterpolantGenerator
from arlib.llm.interpolant.prompts import get_available_prompt_types, get_prompt_description
import z3


def demo_prompt_types():
    """Demonstrate different prompt types for interpolant generation."""

    # Example problem: A says x > 6 and y = x + 1, B says y <= 4
    # Shared variable: y
    # Expected interpolant: y <= 7 (since x > 6 implies y > 7, and y <= 4 contradicts this)

    A = [
        "(declare-fun x () Int)",
        "(declare-fun y () Int)",
        "(assert (> x 6))",
        "(assert (= y (+ x 1)))"
    ]
    B = [
        "(declare-fun y () Int)",
        "(assert (<= y 4))"
    ]

    print("=== Craig Interpolant Generation Demo ===")
    print(f"Problem: A = {A}")
    print(f"Problem: B = {B}")
    print("Shared variable: y")
    print("Expected interpolant: y <= 7 (since x > 6 implies y > 7, and y <= 4 contradicts this)")
    print()

    # Show available prompt types
    print("Available prompt types:")
    for prompt_type in get_available_prompt_types():
        print(f"  - {prompt_type}: {get_prompt_description(prompt_type)}")
    print()

    # Test each prompt type
    for prompt_type in get_available_prompt_types():
        print(f"=== Testing {prompt_type.upper()} prompt ===")
        try:
            gen = LLMInterpolantGenerator(prompt_type=prompt_type)
            res = gen.generate(A, B, max_attempts=1)  # Single attempt for demo

            print(f"Generated interpolant: {res.raw_text}")
            print(f"Valid A => I: {res.valid_A_implies_I}")
            print(f"Valid I ∧ B unsat: {res.unsat_I_and_B}")
            print(f"Success: {res.interpolant is not None}")

        except Exception as e:
            print(f"Error with {prompt_type} prompt: {e}")

        print("-" * 50)


def demo_prompt_content():
    """Show the actual content of different prompt types."""
    from arlib.llm.interpolant.prompts import mk_interpolant_prompt_with_type

    # Convert to Z3 expressions for prompt generation
    A_z3 = [
        z3.parse_smt2_string("(declare-fun x () Int) (assert (> x 6)) (assert (= y (+ x 1)))")[0]
    ]
    B_z3 = [
        z3.parse_smt2_string("(declare-fun y () Int) (assert (<= y 4))")[0]
    ]

    print("\n=== Prompt Content Examples ===")

    for prompt_type in ["basic", "cot", "fewshot", "structured"]:
        print(f"\n--- {prompt_type.upper()} PROMPT ---")
        prompt = mk_interpolant_prompt_with_type(A_z3, B_z3, prompt_type)
        print(prompt)
        print("\n" + "="*80)


if __name__ == "__main__":
    demo_prompt_types()
    demo_prompt_content()
