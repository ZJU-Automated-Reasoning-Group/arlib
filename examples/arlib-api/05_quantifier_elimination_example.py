#!/usr/bin/env python3
"""
Quantifier Elimination Example

This example demonstrates arlib's quantifier elimination capabilities
for different theories and logics.
"""

import z3
from arlib.quant.qe.qe_lme import qelim_exists_lme


def basic_qe_example():
    """Demonstrate basic quantifier elimination."""
    print("=== Basic QE Example ===")
    
    x, y = z3.Reals('x y')
    
    # Formula: ∃x. (x + y > 0 ∧ x < 5)
    formula = z3.And(x + y > 0, x < 5)
    
    try:
        result = qelim_exists_lme(formula, [x])
        print(f"Original: ∃x. {formula}")
        print(f"QE result: {result}")
    except Exception as e:
        print("QE computation failed")


def linear_arithmetic_qe():
    """QE for linear arithmetic."""
    print("\n=== Linear Arithmetic QE ===")
    
    x, y, z = z3.Reals('x y z')
    
    # Formula: ∃x. (2x + y ≤ 10 ∧ x ≥ 0 ∧ x + z ≥ 5)
    formula = z3.And(2*x + y <= 10, x >= 0, x + z >= 5)
    
    try:
        result = qelim_exists_lme(formula, [x])
        print(f"QE result computed")
    except Exception as e:
        print("Linear arithmetic QE failed")


def z3_builtin_qe():
    """Compare with Z3's built-in QE."""
    print("\n=== Z3 Built-in QE ===")
    
    x, y = z3.Reals('x y')
    formula = z3.Exists([x], z3.And(x + y > 0, x < 5))
    
    # Use Z3's tactic
    qe_tactic = z3.Tactic('qe')
    result = qe_tactic(formula)
    
    print(f"Z3 QE result: {result}")


def main():
    """Run all QE examples."""
    print("Quantifier Elimination Examples - arlib's QE")
    print("=" * 45)
    
    basic_qe_example()
    linear_arithmetic_qe()
    z3_builtin_qe()
    
    print("\nQE examples completed!")


if __name__ == "__main__":
    main() 