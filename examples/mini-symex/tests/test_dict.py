#!/usr/bin/env python3
"""
Test file for dictionary operations in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

def test_dict_operations():
    """Test dictionary operations"""
    print("\n=== Testing Dictionary Operations ===")
    
    d = mk_dict("d")
    
    # Dictionary assignment and retrieval
    d["key1"] = 10
    d["key2"] = 20
    
    if "key1" in d:
        if d["key1"] == 10:
            print(f"d['key1'] == 10: d={d}")
    
    # Dictionary get with default
    val = d.get("key3", 30)
    if val == 30:
        print(f"d.get('key3', 30) == 30: d={d}")
    
    # Dictionary update
    d["key1"] = 15
    if d["key1"] == 15:
        print(f"d['key1'] update to 15 successful: d={d}")

if __name__ == "__main__":
    print("=== Running Dictionary Operations Test ===")
    concolic(test_dict_operations, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 