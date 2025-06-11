#!/usr/bin/env python3
"""
AllSMT Example - Enumerating All Satisfying Models

This example demonstrates how to use arlib's AllSMT functionality to enumerate
all satisfying models of SMT formulas. We show examples with different solvers
and various types of formulas.

Since arlib doesn't have an abstraction layer for SMT objects, we use Z3's
Python API objects directly, but arlib provides much richer functionalities
for model enumeration.
"""

import z3
from arlib.allsmt import create_allsmt_solver


def boolean_formula_example():
    """Example with Boolean formulas - finite model space."""
    print("=== Boolean Formula Example ===")
    
    a, b, c = z3.Bools('a b c')
    formula = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), c), z3.Or(b, z3.Not(c)))
    
    solver = create_allsmt_solver("z3")
    models = solver.solve(formula, [a, b, c], model_limit=10)
    print(f"Found {solver.get_model_count()} models")
    solver.print_models(verbose=False)


def integer_formula_example():
    """Example with integer formulas - potentially infinite model space."""
    print("\n=== Integer Formula Example ===")
    
    x, y = z3.Ints('x y')
    formula = z3.And(x + y == 5, x >= 0, y >= 0, x <= 3, y <= 3)
    
    solver = create_allsmt_solver("z3")
    models = solver.solve(formula, [x, y], model_limit=10)
    print(f"Found {solver.get_model_count()} models")
    solver.print_models(verbose=False)


def mixed_formula_example():
    """Example with mixed Boolean and integer variables."""
    print("\n=== Mixed Formula Example ===")
    
    x, y = z3.Ints('x y')
    p, q = z3.Bools('p q')
    formula = z3.And(z3.Implies(p, x > 0), z3.Implies(q, y > 0), 
                     z3.Or(p, q), x + y <= 3, x >= 0, y >= 0)
    
    solver = create_allsmt_solver("z3")
    models = solver.solve(formula, [x, y, p, q], model_limit=15)
    print(f"Found {solver.get_model_count()} models")
    solver.print_models(verbose=False)


def compare_solvers_example():
    """Compare different AllSMT solver backends."""
    print("\n=== Solver Comparison ===")
    
    a, b = z3.Bools('a b')
    formula = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), z3.Not(b)))
    
    # Test Z3 solver
    z3_solver = create_allsmt_solver("z3")
    z3_solver.solve(formula, [a, b], model_limit=5)
    print(f"Z3 found {z3_solver.get_model_count()} models")
    
    # Test PySMT solver (if available)
    try:
        pysmt_solver = create_allsmt_solver("pysmt")
        pysmt_solver.solve(formula, [a, b], model_limit=5)
        print(f"PySMT found {pysmt_solver.get_model_count()} models")
    except Exception as e:
        print(f"PySMT solver not available")


def real_arithmetic_example():
    """Example with real arithmetic - infinite model space."""
    print("\n=== Real Arithmetic Example ===")
    
    x, y = z3.Reals('x y')
    formula = z3.And(x*x + y*y <= 1, x >= 0, y >= 0, x <= 1, y <= 1)
    
    solver = create_allsmt_solver("z3")
    models = solver.solve(formula, [x, y], model_limit=3)
    print(f"Found {solver.get_model_count()} sample models (infinite space)")
    solver.print_models(verbose=False)


def main():
    """Run all AllSMT examples."""
    print("AllSMT Examples - arlib's Model Enumeration")
    print("=" * 45)
    
    boolean_formula_example()
    integer_formula_example()
    mixed_formula_example()
    compare_solvers_example()
    real_arithmetic_example()
    
    print("\nAllSMT examples completed!")


if __name__ == "__main__":
    main()