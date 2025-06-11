#!/usr/bin/env python3
"""Model Counting Example - arlib's counting capabilities for Boolean and arithmetic formulas."""

import z3
from arlib.counting.bool.z3py_expr_counting import count_z3_solutions, count_z3_models_by_enumeration
from arlib.counting.arith_counting_latte import count_lia_models
from arlib.allsmt.bool_enumeration import count_models


def boolean_counting_example():
    """Demonstrate Boolean model counting."""
    print("=== Boolean Model Counting ===")
    
    a, b, c = z3.Bools('a b c')
    formula = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), c))
    
    try:
        count1 = count_z3_solutions(formula)
        count2 = count_z3_models_by_enumeration(formula)
        print(f"Formula: {formula}")
        print(f"DIMACS count: {count1}, Enumeration count: {count2}")
    except Exception as e:
        print(f"Boolean counting failed: {e}")


def enumeration_methods_comparison():
    """Compare different counting methods."""
    print("\n=== Method Comparison ===")
    
    p, q = z3.Bools('p q')
    formula = z3.Or(z3.And(p, q), z3.And(z3.Not(p), z3.Not(q)))
    variables = [p, q]
    
    methods = ['solver', 'enum1', 'enum2']
    
    for method in methods:
        try:
            count = count_models(formula, variables, method=method)
            print(f"{method}: {count} models")
        except Exception as e:
            print(f"{method}: failed")


def arithmetic_counting_example():
    """Use arlib's arithmetic model counter."""
    print("\n=== Arithmetic Model Counter ===")
    
    x, y = z3.Ints('x y')
    formula = z3.And(x + y == 5, x >= 0, y >= 0, x <= 3, y <= 3)
    
    try:
        count = count_lia_models(formula)
        print(f"Integer formula model count: {count}")
    except Exception as e:
        print(f"Arithmetic counter failed: {e}")


def main():
    """Run all counting examples."""
    print("Model Counting Examples")
    print("=" * 25)
    
    boolean_counting_example()
    enumeration_methods_comparison()
    arithmetic_counting_example()
    
    print("\nCounting examples completed!")


if __name__ == "__main__":
    main() 