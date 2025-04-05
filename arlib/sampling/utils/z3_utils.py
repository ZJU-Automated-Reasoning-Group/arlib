"""
Z3 utility functions.

This module provides utility functions for working with Z3 formulas.
"""

import z3
from typing import List, Set


def get_vars(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """
    Extract all variables from a Z3 formula.
    
    Args:
        formula: The Z3 formula to extract variables from
        
    Returns:
        List of variables in the formula
    """
    vars_set = set()

    def collect(expr):
        if z3.is_const(expr) and not z3.is_true(expr) and not z3.is_false(expr):
            vars_set.add(expr)
        for child in expr.children():
            collect(child)

    collect(formula)
    return list(vars_set)


def is_bool(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a Boolean variable.
    
    Args:
        expr: The Z3 expression to check
        
    Returns:
        True if the expression is a Boolean variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.BoolSort()


def is_bv(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a bit-vector variable.
    
    Args:
        expr: The Z3 expression to check
        
    Returns:
        True if the expression is a bit-vector variable, False otherwise
    """
    return z3.is_const(expr) and z3.is_bv_sort(expr.sort())


def is_int(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is an integer variable.
    
    Args:
        expr: The Z3 expression to check
        
    Returns:
        True if the expression is an integer variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.IntSort()


def is_real(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a real variable.
    
    Args:
        expr: The Z3 expression to check
        
    Returns:
        True if the expression is a real variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.RealSort()
