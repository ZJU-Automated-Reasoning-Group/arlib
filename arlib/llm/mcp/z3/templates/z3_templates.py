"""
Z3 template functions for common quantifier patterns and constraints.
"""

from z3 import And, ArrayRef, BoolRef, ExprRef, Or, PbEq, PbGe, PbLe

def array_is_sorted(arr: ArrayRef, size: int, strict: bool = False) -> BoolRef:
    """
    Create a constraint that an array is sorted (for Z3 Array objects).

    Args:
        arr: Z3 Array object
        size: Number of elements to check
        strict: If True, uses strict inequality (<) instead of (<=)

    Returns:
        Z3 boolean constraint
    """
    if strict:
        return And([arr[i] < arr[i + 1] for i in range(size - 1)])
    else:
        return And([arr[i] <= arr[i + 1] for i in range(size - 1)])


def all_distinct(arr: ArrayRef, size: int) -> BoolRef:
    """
    Create a constraint that all elements in an array are distinct.

    Args:
        arr: Z3 Array object
        size: Number of elements

    Returns:
        Z3 boolean constraint
    """
    constraints = []
    for i in range(size):
        for j in range(i + 1, size):
            constraints.append(arr[i] != arr[j])
    return And(constraints)


def array_contains(arr: ArrayRef, size: int, value: ExprRef) -> BoolRef:
    """
    Create a constraint that an array contains a specific value.

    Args:
        arr: Z3 Array object
        size: Number of elements to check
        value: The value that must be present

    Returns:
        Z3 boolean constraint
    """
    return Or([arr[i] == value for i in range(size)])


def exactly_k(bool_vars: list[BoolRef], k: int) -> BoolRef:
    """
    Create a constraint that exactly k boolean variables are true.

    Args:
        bool_vars: List of Z3 boolean variables
        k: Number of variables that must be true

    Returns:
        Z3 boolean constraint
    """
    return PbEq([(var, 1) for var in bool_vars], k)


def at_most_k(bool_vars: list[BoolRef], k: int) -> BoolRef:
    """
    Create a constraint that at most k boolean variables are true.

    Args:
        bool_vars: List of Z3 boolean variables
        k: Maximum number of variables that can be true

    Returns:
        Z3 boolean constraint
    """
    return PbLe([(var, 1) for var in bool_vars], k)


def at_least_k(bool_vars: list[BoolRef], k: int) -> BoolRef:
    """
    Create a constraint that at least k boolean variables are true.

    Args:
        bool_vars: List of Z3 boolean variables
        k: Minimum number of variables that must be true

    Returns:
        Z3 boolean constraint
    """
    return PbGe([(var, 1) for var in bool_vars], k)


def function_is_injective(func: ArrayRef, domain_size: int, codomain_size: int) -> BoolRef:
    """
    Create a constraint that a function (represented as an array) is injective.

    Args:
        func: Z3 Array representing the function
        domain_size: Size of the domain
        codomain_size: Size of the codomain

    Returns:
        Z3 boolean constraint
    """
    constraints = []
    # For all i, j in domain: if i != j then func[i] != func[j]
    for i in range(domain_size):
        for j in range(i + 1, domain_size):
            constraints.append(func[i] != func[j])
    return And(constraints)


def function_is_surjective(func: ArrayRef, domain_size: int, codomain_size: int) -> BoolRef:
    """
    Create a constraint that a function (represented as an array) is surjective.

    Args:
        func: Z3 Array representing the function
        domain_size: Size of the domain
        codomain_size: Size of the codomain

    Returns:
        Z3 boolean constraint
    """
    constraints = []
    # For all y in codomain: exists x in domain such that func[x] = y
    for y in range(codomain_size):
        constraints.append(Or([func[x] == y for x in range(domain_size)]))
    return And(constraints)
