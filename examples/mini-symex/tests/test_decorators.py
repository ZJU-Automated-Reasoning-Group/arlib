#!/usr/bin/env python3
"""
Test file for decorators in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_decorators():
    """Test decorators"""
    print("\n=== Testing Decorators ===")
    
    # Define a decorator
    def double_result(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result * 2
        return wrapper
    
    # Apply the decorator
    @double_result
    def add(a, b):
        return a + b
    
    x = mk_int("x")
    y = mk_int("y")
    
    result = add(x, y)
    
    if result > 20:
        print(f"Decorated add(x, y) > 20: x={x}, y={y}, result={result}")

if __name__ == "__main__":
    print("=== Running Decorators Test ===")
    concolic(test_decorators, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 