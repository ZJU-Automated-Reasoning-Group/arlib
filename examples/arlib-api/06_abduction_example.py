#!/usr/bin/env python3
"""
Abduction Example

This example demonstrates arlib's abduction capabilities.
Abduction finds explanations: given precondition P and postcondition Q,
find hypothesis H such that P ∧ H ⊨ Q.
"""

import z3
from arlib.abduction.qe_abduct import qe_abduce
from arlib.abduction.dillig_abduct import dillig_abduce


def basic_abduction_example():
    """Demonstrate basic abduction."""
    print("=== Basic Abduction ===")
    
    x, y = z3.Reals('x y')
    pre_cond = z3.And(x <= 0, y > 1)
    post_cond = x + y <= 5
    
    try:
        result = qe_abduce(pre_cond, post_cond)
        print(f"Precondition: {pre_cond}")
        print(f"Postcondition: {post_cond}")
        print(f"QE result: {result}")
    except Exception as e:
        print(f"QE abduction failed: {e}")


def dillig_abduction_example():
    """Demonstrate Dillig's abduction algorithm."""
    print("\n=== Dillig Abduction ===")
    
    x, y = z3.Ints('x y')
    pre_cond = z3.And(x >= 0, y >= 0)
    post_cond = x + y >= 5
    
    try:
        result = dillig_abduce(pre_cond, post_cond)
        print(f"Precondition: {pre_cond}")
        print(f"Postcondition: {post_cond}")
        if result is not None:
            print(f"Dillig result: {result}")
        else:
            print("Dillig result: No abduction found")
    except Exception as e:
        print(f"Dillig abduction failed: {e}")


def comparison_example():
    """Compare different abduction methods."""
    print("\n=== Method Comparison ===")
    
    x, y = z3.Reals('x y')
    pre_cond = z3.And(x >= 0, y >= 0)
    post_cond = x + y >= 5
    
    methods = [
        ("QE-based", qe_abduce),
        ("Dillig", dillig_abduce)
    ]
    
    for name, method in methods:
        try:
            result = method(pre_cond, post_cond)
            if result is not None:
                print(f"{name}: {result}")
            else:
                print(f"{name}: No result")
        except Exception as e:
            print(f"{name}: Failed - {e}")


def main():
    """Run all abduction examples."""
    print("Abduction Examples - arlib's Abduction")
    print("=" * 40)
    
    basic_abduction_example()
    dillig_abduction_example()
    comparison_example()
    
    print("\nAbduction examples completed!")


if __name__ == "__main__":
    main() 