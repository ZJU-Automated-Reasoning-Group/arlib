#!/usr/bin/env python3
"""
Test file demonstrating the use of state and path controls in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_state_control():
    """Test function with multiple paths to demonstrate state control"""
    print("\n=== Testing State Control ===")
    
    x = mk_int("x")
    y = mk_int("y")
    
    # Create multiple paths to explore
    if x > 0:
        if y > 0:
            print(f"Path 1: x > 0, y > 0: x={x}, y={y}")
        else:
            print(f"Path 2: x > 0, y <= 0: x={x}, y={y}")
    else:
        if y > 0:
            print(f"Path 3: x <= 0, y > 0: x={x}, y={y}")
        else:
            print(f"Path 4: x <= 0, y <= 0: x={x}, y={y}")
    
    # Create more paths with nested conditions
    if x % 2 == 0:
        if y % 2 == 0:
            print(f"Path 5: x is even, y is even: x={x}, y={y}")
        else:
            print(f"Path 6: x is even, y is odd: x={x}, y={y}")
    else:
        if y % 2 == 0:
            print(f"Path 7: x is odd, y is even: x={x}, y={y}")
        else:
            print(f"Path 8: x is odd, y is odd: x={x}, y={y}")

def run_with_limits():
    """Run the test with different state and path limits"""
    print("\n=== Running with unlimited states and paths ===")
    crashes, stats = concolic(test_state_control, debug=True, exit_on_err=False)
    print(f"\nExploration completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running with max_states=2 ===")
    crashes, stats = concolic(test_state_control, debug=True, exit_on_err=False, max_states=2)
    print(f"\nExploration completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running with max_paths=3 ===")
    crashes, stats = concolic(test_state_control, debug=True, exit_on_err=False, max_paths=3)
    print(f"\nExploration completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running with max_states=3 and max_paths=5 ===")
    crashes, stats = concolic(test_state_control, debug=True, exit_on_err=False, max_states=3, max_paths=5)
    print(f"\nExploration completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")

if __name__ == "__main__":
    print("=== Running State Control Test ===")
    run_with_limits()
    print("\n=== Test Completed ===") 