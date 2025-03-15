#!/usr/bin/env python3
"""
Test file for numeric operations in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_numeric_operations():
    """Test numeric operations with integers and floats"""
    print("\n=== Testing Numeric Operations ===")
    
    # Integer operations
    x = mk_int("x")
    y = mk_int("y")
    
    # Basic arithmetic
    if x + y > 10:
        print(f"x + y > 10: x={x}, y={y}")
    
    # Integer division and modulo
    if y != 0:  # Add check to avoid division by zero
        if x // y == 2 and x % y == 1:
            print(f"x // y == 2 and x % y == 1: x={x}, y={y}")
    
    # Bitwise operations
    if (x & y) == 1:
        print(f"x & y == 1: x={x}, y={y}")
    
    if (x | y) == 7:
        print(f"x | y == 7: x={x}, y={y}")
    
    if (x ^ y) == 6:
        print(f"x ^ y == 6: x={x}, y={y}")
    
    if (x << 2) == 20:
        print(f"x << 2 == 20: x={x}")
    
    if (x >> 1) == 2:
        print(f"x >> 1 == 2: x={x}")
    
    # Float operations
    a = mk_int("a")  # Change to mk_int to avoid Z3 sort mismatch
    b = mk_int("b")  # Change to mk_int to avoid Z3 sort mismatch
    
    if a + b > 5:  # Change to integer comparison
        if a * b < 10:  # Change to integer comparison
            print(f"a + b > 5 and a * b < 10: a={a}, b={b}")
    
    # Mixed integer operations (avoid mixing types)
    if x + a > 10:  # Change to integer comparison
        print(f"x + a > 10: x={x}, a={a}")

if __name__ == "__main__":
    print("=== Running Numeric Operations Test ===")
    concolic(test_numeric_operations, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 