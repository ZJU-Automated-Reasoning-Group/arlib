#!/usr/bin/env python3
"""
Test file for function calls and parameters in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_functions():
    """Test function calls and parameters"""
    print("\n=== Testing Functions ===")
    
    x = mk_int("x")
    y = mk_int("y")
    
    # Function with positional arguments
    def add(a, b):
        return a + b
    
    result = add(x, y)
    if result > 10:
        print(f"add(x, y) > 10: x={x}, y={y}, result={result}")
    
    # Function with default arguments
    def multiply(a, b=2):
        return a * b
    
    result = multiply(x)
    if result > 15:
        print(f"multiply(x) > 15: x={x}, result={result}")
    
    # Function with keyword arguments
    result = multiply(a=x, b=y)
    if result > 20:
        print(f"multiply(a=x, b=y) > 20: x={x}, y={y}, result={result}")

if __name__ == "__main__":
    print("=== Running Functions Test ===")
    concolic(test_functions, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 