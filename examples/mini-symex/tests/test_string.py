#!/usr/bin/env python3
"""
Test file for string operations in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_string_operations():
    """Test string operations"""
    print("\n=== Testing String Operations ===")
    
    s1 = mk_str("s1")
    s2 = mk_str("s2")
    
    # String concatenation
    concat = s1 + s2
    if concat == "hello world":
        print(f"s1 + s2 == 'hello world': s1='{s1}', s2='{s2}'")
    
    # String methods
    if s1.startswith("py"):
        print(f"s1 starts with 'py': s1='{s1}'")
    
    if s2.endswith("thon"):
        print(f"s2 ends with 'thon': s2='{s2}'")
    
    # String contains
    if "th" in s1:
        print(f"'th' in s1: s1='{s1}'")
    
    # String indexing and slicing
    if len(s1) > 3:
        first_char = s1[0]
        slice_str = s1[1:3]
        if first_char == 'p' and slice_str == "yt":
            print(f"s1[0] == 'p' and s1[1:3] == 'yt': s1='{s1}'")

if __name__ == "__main__":
    print("=== Running String Operations Test ===")
    concolic(test_string_operations, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 