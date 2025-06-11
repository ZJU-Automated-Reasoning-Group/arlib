#!/usr/bin/env python3
"""SyGuS examples using arlib's synthesis capabilities"""

import z3
from arlib.sygus.sygus_pbe import StringSyGuSPBE
from arlib.utils.z3_plus_smtlib_solver import Z3SolverPlus

def string_synthesis():
    print("=== String Synthesis ===")
    try:
        synthesizer = StringSyGuSPBE(debug=False)
        
        examples = [("hello", "world", "helloworld"), ("syn", "thesis", "synthesis")]
        result = synthesizer.synthesize_concat_function(examples, "concat_func")
        if result:
            print(f"Concat: {result}")
        
        transform_examples = [("hello", "hel"), ("synthesis", "syn")]
        result = synthesizer.synthesize_string_transformer(transform_examples, "first_three")
        if result:
            print(f"Transform: {result}")
            
    except Exception as e:
        print(f"Failed: {e}")

def function_synthesis():
    print("\n=== Function Synthesis ===")
    try:
        solver = Z3SolverPlus(debug=False)
        
        max_func = z3.Function("max", z3.IntSort(), z3.IntSort(), z3.IntSort())
        x, y = z3.Ints("x y")
        
        constraints = [
            max_func(x, y) >= x,
            max_func(x, y) >= y,
            z3.Or(max_func(x, y) == x, max_func(x, y) == y)
        ]
        
        result = solver.sygus([max_func], constraints, [x, y], logic="LIA")
        if result and "define-fun" in result:
            print(f"Max function: {result}")
            
    except Exception as e:
        print(f"Failed: {e}")

def bitvector_synthesis():
    print("\n=== BitVector Synthesis ===")
    try:
        solver = Z3SolverPlus(debug=False)
        
        bv_func = z3.Function("bv_func", z3.BitVecSort(4), z3.BitVecSort(4))
        bv_x = z3.BitVec("x", 4)
        
        constraints = [
            bv_func(z3.BitVecVal(1, 4)) == z3.BitVecVal(8, 4),
            bv_func(z3.BitVecVal(2, 4)) == z3.BitVecVal(1, 4),
            bv_func(z3.BitVecVal(4, 4)) == z3.BitVecVal(2, 4)
        ]
        
        result = solver.sygus([bv_func], constraints, [bv_x], logic="BV")
        if result and "define-fun" in result:
            print(f"BitVector function: {result}")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    string_synthesis()
    function_synthesis()
    bitvector_synthesis() 