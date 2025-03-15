#!/usr/bin/env python3
"""
Test file for loop constructs in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_loops():
    """Test loop constructs"""
    print("\n=== Testing Loops ===")
    
    n = mk_int("n")
    
    # Symbolic range loop
    print("Testing symbolic range:")
    sum_val = 0
    for i in symbolic_range(n):
        sum_val += i
    
    if sum_val > 10:
        print(f"Sum from 0 to {n} is greater than 10: sum={sum_val}")
    else:
        print(f"Sum from 0 to {n} is not greater than 10: sum={sum_val}")
    
    # While loop with symbolic condition
    print("Testing while loop:")
    x = mk_int("x")
    count = 0
    
    # Limit iterations to avoid infinite loops
    max_iter = 5
    while x > 0 and count < max_iter:
        x -= 1
        count += 1
    
    print(f"While loop executed {count} times with initial x={x+count}")

if __name__ == "__main__":
    print("=== Running Loops Test ===")
    concolic(test_loops, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 