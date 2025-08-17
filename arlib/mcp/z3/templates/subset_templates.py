"""
Subset-related template functions for Z3.
"""

from collections.abc import Callable
from itertools import combinations
from typing import TypeVar, Optional, List

import z3
from z3 import Bool, ExprRef, Solver

# Type variable for generic item types
T = TypeVar("T")


def smallest_subset_with_property(
    items: List[T],
    property_check_func: Callable[[List[T]], bool],
    min_size: int = 1,
    max_size: Optional[int] = None,
) -> Optional[List[T]]:
    """
    Find the smallest subset of items that satisfies a given property.

    This function uses a brute-force approach to find the minimal subset.
    For large item sets, consider using Z3 optimization instead.

    Args:
        items: List of items to choose from
        property_check_func: Function that returns True if a subset has the desired property
        min_size: Minimum size of subset to consider
        max_size: Maximum size of subset to consider (defaults to len(items))

    Returns:
        The smallest subset that satisfies the property, or None if no such subset exists
    """
    if max_size is None:
        max_size = len(items)

    # Try subsets of increasing size
    for size in range(min_size, max_size + 1):
        for subset in combinations(items, size):
            subset_list = list(subset)
            if property_check_func(subset_list):
                return subset_list

    return None


def subset_selection_template(
    items: List[str],
    weights: Optional[List[int]] = None,
    capacity: int = 10
    ) -> tuple[Solver, dict[str, ExprRef]]:
    """
    Template for subset selection problems (like knapsack).

    Args:
        items: List of item names
        weights: List of item weights (defaults to 1 for each item)
        capacity: Maximum total weight

    Returns:
        Tuple of (solver, variables_dict)
    """
    if weights is None:
        weights = [1] * len(items)

    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length")

    # Create boolean variables for each item (selected or not)
    selected = [Bool(f"select_{item}") for item in items]

    # Create solver
    solver = Solver()

    # Capacity constraint: sum of weights <= capacity
    total_weight = z3.Sum([z3.If(selected[i], weights[i], 0) for i in range(len(items))])
    solver.add(total_weight <= capacity)

    # Create variables dictionary
    variables = {f"select_{items[i]}": selected[i] for i in range(len(items))}
    variables["total_weight"] = total_weight

    return solver, variables
