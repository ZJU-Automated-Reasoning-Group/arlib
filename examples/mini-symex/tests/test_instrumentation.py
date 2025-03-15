#!/usr/bin/env python3
"""
Test file for automatic instrumentation in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def complex_function(a, b, c):
    """A complex function with multiple paths and types"""
    result = 0
    
    if isinstance(a, int) and a > 10:
        result += a * 2
    
    if isinstance(b, str) and len(b) > 3:
        if "test" in b:
            result += 10
        else:
            result += 5
    
    if isinstance(c, list) and len(c) > 0:
        for item in c:
            if isinstance(item, int) and item % 2 == 0:
                result += item
    
    return result

def test_instrumentation():
    """Test automatic instrumentation"""
    print("\n=== Testing Automatic Instrumentation ===")
    
    # This will automatically create symbolic variables
    instrumented = instrument_function(complex_function)
    
    # Call with concrete values
    result = instrumented(15, "testing", [2, 4, 6])
    
    print(f"Result of instrumented function: {result}")

if __name__ == "__main__":
    print("=== Running Automatic Instrumentation Test ===")
    concolic(test_instrumentation, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 