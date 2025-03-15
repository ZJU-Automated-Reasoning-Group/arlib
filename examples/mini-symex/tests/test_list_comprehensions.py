#!/usr/bin/env python3
"""
Test file for list comprehensions in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_list_comprehensions():
    """Test list comprehensions"""
    print("\n=== Testing List Comprehensions ===")
    
    x = mk_int("x")
    
    # Create a list using list comprehension
    lst = [i * x for i in range(1, 6)]
    
    if sum(lst) > 50:
        print(f"Sum of [i * x for i in range(1, 6)] > 50: x={x}, lst={lst}")

if __name__ == "__main__":
    print("=== Running List Comprehensions Test ===")
    concolic(test_list_comprehensions, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 