#!/usr/bin/env python3
"""
Test file for conditional statements in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_conditional_statements():
    """Test conditional statements"""
    print("\n=== Testing Conditional Statements ===")
    
    x = mk_int("x")
    y = mk_int("y")
    
    # Simple if-else
    if x > y:
        print(f"x > y: x={x}, y={y}")
    else:
        print(f"x <= y: x={x}, y={y}")
    
    # Nested if-elif-else
    if x > 10:
        if y > 5:
            print(f"x > 10 and y > 5: x={x}, y={y}")
        else:
            print(f"x > 10 and y <= 5: x={x}, y={y}")
    elif x > 5:
        print(f"5 < x <= 10: x={x}")
    else:
        print(f"x <= 5: x={x}")

if __name__ == "__main__":
    print("=== Running Conditional Statements Test ===")
    concolic(test_conditional_statements, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 