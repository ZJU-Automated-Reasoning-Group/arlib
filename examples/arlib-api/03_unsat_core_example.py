#!/usr/bin/env python3
"""Unsat Core Example - arlib's unsat core computation for debugging unsatisfiable formulas."""

import z3
from arlib.unsat_core.marco import SubsetSolver, MapSolver, enumerate_sets


def basic_unsat_core_example():
    """Demonstrate basic unsat core computation using Z3."""
    print("=== Basic Unsat Core ===")
    
    x, y, z = z3.Ints('x y z')
    constraints = [x + y <= 5, x >= 3, y >= 4, x + y >= 10, z >= 0]
    
    solver = z3.Solver()
    for constraint in constraints:
        solver.add(constraint)
    
    result = solver.check()
    print(f"Satisfiability: {result}")
    
    if result == z3.unsat:
        # For unsat core, we need to use assumptions or assertions with names
        solver2 = z3.Solver()
        assumptions = []
        for i, constraint in enumerate(constraints):
            assumption = z3.Bool(f"c{i}")
            solver2.add(z3.Implies(assumption, constraint))
            assumptions.append(assumption)
        
        result2 = solver2.check(assumptions)
        if result2 == z3.unsat:
            core = solver2.unsat_core()
            print(f"Z3 Unsat Core: {len(core)} constraints: {core}")


def named_constraints_example():
    """Demonstrate unsat core with named constraints for better debugging."""
    print("\n=== Named Constraints ===")
    
    a, b, c = z3.Bools('a b c')
    c1, c2, c3, c4, c5 = z3.Bools('constraint_1 constraint_2 constraint_3 constraint_4 constraint_5')
    
    solver = z3.Solver()
    solver.add(z3.Implies(c1, a))
    solver.add(z3.Implies(c2, z3.Not(a)))
    solver.add(z3.Implies(c3, z3.Or(a, b)))
    solver.add(z3.Implies(c4, z3.Not(b)))
    solver.add(z3.Implies(c5, c))
    
    assumptions = [c1, c2, c3, c4, c5]
    result = solver.check(assumptions)
    print(f"Satisfiability: {result}")
    
    if result == z3.unsat:
        core = solver.unsat_core()
        print(f"Unsat Core: {len(core)} named constraints: {core}")


def marco_mus_example():
    """Demonstrate MUS enumeration using Marco algorithm."""
    print("\n=== Marco MUS Enumeration ===")
    
    x, y = z3.Reals('x y')
    constraints = [x > 2, x < 1, y >= 0, y < 0]
    
    try:
        csolver = SubsetSolver(constraints)
        msolver = MapSolver(n=csolver.n)
        
        mus_count = 0
        mss_count = 0
        
        for orig, lits in enumerate_sets(csolver, msolver):
            if orig == "MUS":
                mus_count += 1
                print(f"MUS {mus_count}: {len(lits)} constraints")
            elif orig == "MSS":
                mss_count += 1
                print(f"MSS {mss_count}: {len(lits)} constraints")
        
        print(f"Total: {mus_count} MUS, {mss_count} MSS")
    except Exception as e:
        print(f"Marco enumeration failed: {e}")


def simple_core_analysis():
    """Simple unsat core analysis for practical debugging."""
    print("\n=== Simple Core Analysis ===")
    
    x, y, z = z3.Ints('x y z')
    
    # Create conflicting constraints with assumptions
    constraints = [
        ("lower_bound_x", x >= 10),
        ("upper_bound_x", x <= 5),
        ("lower_bound_y", y >= 8),
        ("upper_bound_y", y <= 3),
        ("sum_constraint", x + y == 15)
    ]
    
    solver = z3.Solver()
    assumptions = []
    
    for name, constraint in constraints:
        assumption = z3.Bool(name)
        solver.add(z3.Implies(assumption, constraint))
        assumptions.append(assumption)
    
    result = solver.check(assumptions)
    print(f"Satisfiability: {result}")
    
    if result == z3.unsat:
        core = solver.unsat_core()
        print(f"Conflicting constraints: {list(core)}")


def main():
    """Run all unsat core examples."""
    print("Unsat Core Examples")
    print("=" * 20)
    
    basic_unsat_core_example()
    named_constraints_example()
    marco_mus_example()
    simple_core_analysis()
    
    print("\nUnsat core examples completed!")


if __name__ == "__main__":
    main() 