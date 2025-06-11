#!/usr/bin/env python3
"""Unification examples using arlib's term unification capabilities"""

import z3
from arlib.itp.utils import unify, pmatch, antipattern
from arlib.unification.core import unify as core_unify, reify
from arlib.unification.variable import var

def z3_unification():
    print("=== Z3 Unification ===")
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    
    term1 = f(x)
    term2 = f(z3.IntVal(5))
    
    subst = unify([x], term1, term2)
    if subst:
        print(f"Unify {term1} with {term2}: {subst}")
    
    term1 = f(x)
    term2 = f(y)
    subst = unify([x, y], term1, term2)
    if subst:
        print(f"Unify {term1} with {term2}: {subst}")

def pattern_matching():
    print("\n=== Pattern Matching ===")
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    
    pattern = f(x)
    term = f(z3.IntVal(42))
    
    match = pmatch([x], pattern, term)
    if match:
        print(f"Match {pattern} against {term}: {match}")

def anti_unification():
    print("\n=== Anti-Unification ===")
    try:
        a, b, c, d = z3.Ints("a b c d")
        f = z3.Function("f", z3.IntSort(), z3.IntSort(), z3.IntSort())
        
        term1 = f(a, b)
        term2 = f(c, d)
        
        pattern_vars, pattern = antipattern([term1, term2])
        print(f"Generalize {term1}, {term2}: {pattern}")
        
    except Exception as e:
        print(f"Failed: {e}")

def core_unification():
    print("\n=== Core Unification ===")
    x = var('x')
    y = var('y')
    
    term1 = (1, x, (3, y))
    term2 = (1, 2, (3, 4))
    
    subst = core_unify(term1, term2)
    if subst:
        print(f"Unify {term1} with {term2}: {subst}")
        result = reify(term1, subst)
        print(f"Result: {result}")

if __name__ == "__main__":
    z3_unification()
    pattern_matching()
    anti_unification()
    core_unification() 