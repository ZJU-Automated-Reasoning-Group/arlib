#!/usr/bin/env python3
"""
Test file for closures and nested functions in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_closures():
    """Test closures and nested functions"""
    print("\n=== Testing Closures ===")
    
    x = mk_int("x")
    
    def outer_function(a):
        # This is a closure that captures the variable 'a'
        def inner_function(b):
            return a + b
        return inner_function
    
    # Create a closure that captures x
    closure = outer_function(x)
    
    # Call the closure with different values
    y = mk_int("y")
    result = closure(y)
    
    if result > 15:
        print(f"closure(y) > 15: x={x}, y={y}, result={result}")

if __name__ == "__main__":
    print("=== Running Closures Test ===")
    concolic(test_closures, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 