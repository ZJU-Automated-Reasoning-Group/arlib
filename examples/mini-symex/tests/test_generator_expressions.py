#!/usr/bin/env python3
"""
Test file for generator expressions in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_generator_expressions():
    """Test generator expressions"""
    print("\n=== Testing Generator Expressions ===")
    
    x = mk_int("x")
    
    # Create a generator expression
    gen = (i * x for i in range(1, 6))
    
    # Convert to list to use it
    lst = list(gen)
    
    if sum(lst) > 50:
        print(f"Sum of (i * x for i in range(1, 6)) > 50: x={x}, lst={lst}")

if __name__ == "__main__":
    print("=== Running Generator Expressions Test ===")
    concolic(test_generator_expressions, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 