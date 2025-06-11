#!/usr/bin/env python3
"""Interpolation examples using arlib's interpolation capabilities"""

import z3
from arlib.bool.interpolant.core_based_itp import BooleanInterpolant
from arlib.bool.interpolant.pysmt_itp import pysmt_binary_itp
from arlib.utils.pysmt_solver import PySMTSolver

def boolean_interpolation():
    print("=== Boolean Interpolation ===")
    x, y = z3.Bools("x y")
    fml_a = z3.And(x, y)
    fml_b = z3.And(z3.Not(x), z3.Not(y))
    
    itp = BooleanInterpolant.compute_itp(fml_a, fml_b, [x, y])
    print(f"Interpolant: {list(itp)}")

def pysmt_interpolation():
    print("\n=== PySMT Interpolation ===")
    try:
        x, y = z3.Bools("x y")
        fml_a = z3.And(x, y)
        fml_b = z3.And(z3.Not(x), z3.Not(y))
        itp = pysmt_binary_itp(fml_a, fml_b)
        print(f"Result: {itp}")
    except Exception as e:
        print(f"Failed: {e}")

def arithmetic_interpolation():
    print("\n=== Arithmetic Interpolation ===")
    try:
        x, y = z3.Ints("x y")
        fml_a = z3.And(x > 0, y > 0)
        fml_b = z3.And(x < 0, y < 0)
        
        solver = PySMTSolver()
        itp = solver.binary_interpolant(fml_a, fml_b, logic="LIA")
        print(f"Result: {itp}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    boolean_interpolation()
    pysmt_interpolation()
    arithmetic_interpolation() 