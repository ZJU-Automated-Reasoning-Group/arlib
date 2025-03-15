#!/usr/bin/env python3
"""
Test file for exception handling in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_exception_handling():
    """Test exception handling"""
    print("\n=== Testing Exception Handling ===")
    
    x = mk_int("x")
    
    def try_block():
        if x == 0:
            raise ValueError("x cannot be zero")
        elif x < 0:
            raise TypeError("x cannot be negative")
        return 100 // x
    
    def handle_value_error(e):
        print(f"Caught ValueError: {e}")
        return 0
    
    def handle_type_error(e):
        print(f"Caught TypeError: {e}")
        return -1
    
    def handle_zero_division(e):
        print(f"Caught ZeroDivisionError: {e}")
        return float('inf')
    
    def finally_block():
        print("Finally block executed")
    
    result = symbolic_try_except(
        try_block,
        [(ValueError, handle_value_error),
         (TypeError, handle_type_error),
         (ZeroDivisionError, handle_zero_division)],
        finally_block
    )
    
    print(f"Result: {result}")

if __name__ == "__main__":
    print("=== Running Exception Handling Test ===")
    concolic(test_exception_handling, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 