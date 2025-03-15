#!/usr/bin/env python3
"""
Test file for lambda functions in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_lambda_functions():
    """Test lambda functions"""
    print("\n=== Testing Lambda Functions ===")
    
    x = mk_int("x")
    y = mk_int("y")
    
    # Add constraints to limit the search space
    if x > -100 and x < 100 and y > -100 and y < 100:
        # Create lambda functions
        add = lambda a, b: a + b
        square = lambda a: a * a
        
        result1 = add(x, y)
        result2 = square(x)
        
        # Simplify the condition to reduce the search space
        if x > 10:
            print(f"x > 10: x={x}, y={y}, add={result1}, square={result2}")
        else:
            print(f"x <= 10: x={x}, y={y}, add={result1}, square={result2}")

if __name__ == "__main__":
    print("=== Running Lambda Functions Test ===")
    concolic(test_lambda_functions, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 