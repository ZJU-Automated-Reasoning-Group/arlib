"""
Core function templates for Z3 models.
"""

from typing import Tuple, Dict

import z3
from z3 import (
        And,
        Exists, ExprRef, ForAll, Implies, Int, Optimize, Solver, sat, unsat
    )

def constraint_satisfaction_template() -> Tuple[Solver, Dict[str, ExprRef]]:
    """
    Template for basic constraint satisfaction problems.

    Returns:
        Tuple of (solver, variables_dict)
    """
    # Define variables
    x = Int("x")
    y = Int("y")

    # Create solver
    solver = Solver()

    # Add constraints (example)
    solver.add(x > 0)
    solver.add(y > 0)
    solver.add(x + y == 10)

    # Return solver and variables
    variables = {"x": x, "y": y}
    return solver, variables


def optimization_template() -> Tuple[z3.Optimize, Dict[str, ExprRef], ExprRef]:
    """
    Template for optimization problems.

    Returns:
        Tuple of (optimizer, variables_dict, objective)
    """
    # Define variables
    x = Int("x")
    y = Int("y")

    # Create optimizer
    optimizer = Optimize()

    # Add constraints
    optimizer.add(x >= 0)
    optimizer.add(y >= 0)
    optimizer.add(x + y <= 10)

    # Define objective
    objective = x + y
    optimizer.maximize(objective)

    # Return optimizer, variables, and objective
    variables = {"x": x, "y": y}
    return optimizer, variables, objective


def array_template(size: int = 5) -> Tuple[Solver, Dict[str, ExprRef]]:
    """
    Template for array-based problems.

    Args:
        size: Size of the array

    Returns:
        Tuple of (solver, variables_dict)
    """
    # Create array of integer variables
    arr = [Int(f"arr_{i}") for i in range(size)]

    # Create solver
    solver = Solver()

    # Example constraints: array elements in range [1, 10]
    for i in range(size):
        solver.add(arr[i] >= 1)
        solver.add(arr[i] <= 10)

    # Example constraint: array is sorted
    for i in range(size - 1):
        solver.add(arr[i] <= arr[i + 1])

    # Create variables dictionary
    variables = {f"arr_{i}": arr[i] for i in range(size)}

    return solver, variables


def quantifier_template() -> Tuple[Solver, Dict[str, ExprRef]]:
    """
    Template for problems involving quantifiers.

    Returns:
        Tuple of (solver, variables_dict)
    """
    # Define function symbol
    f = z3.Function('f', z3.IntSort(), z3.IntSort())

    # Define domain variables
    x = Int("x")
    y = Int("y")

    # Create solver
    solver = Solver()

    # Example: forall x in [0, 5]: f(x) > 0
    solver.add(ForAll([x], Implies(And(x >= 0, x <= 5), f(x) > 0)))

    # Example: exists y: f(y) == 10
    solver.add(Exists([y], f(y) == 10))

    # Create variables dictionary
    variables = {"f": f, "x": x, "y": y}

    return solver, variables


def demo_template() -> Tuple[Solver, Dict[str, ExprRef]]:
    """
    Simple demo template for testing.

    Returns:
        Tuple of (solver, variables_dict)
    """
    # Simple variables
    a = Int("a")
    b = Int("b")

    # Simple solver
    solver = Solver()
    solver.add(a > 0)
    solver.add(b > a)
    solver.add(a + b < 10)

    variables = {"a": a, "b": b}
    return solver, variables
