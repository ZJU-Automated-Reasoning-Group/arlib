#!/usr/bin/env python3
"""
Script to run all the individual test files in the mini-symex framework.
"""

import os
import subprocess
import sys

def run_test(test_file):
    """Run a single test file and return whether it passed"""
    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print(f"{'='*50}")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to the test file
    test_file_path = os.path.join(script_dir, test_file)
    
    result = subprocess.run(['python', test_file_path], capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ {test_file} PASSED")
        return True
    else:
        print(f"\n❌ {test_file} FAILED")
        return False

def main():
    """Run all test files in the tests directory"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all Python files in the directory that start with "test_"
    test_files = [f for f in os.listdir(script_dir) 
                 if f.startswith('test_') and f.endswith('.py') and f != 'test_comprehensive.py']
    
    # Sort the test files for consistent ordering
    test_files.sort()
    
    print(f"Found {len(test_files)} test files to run")
    
    # Run each test file
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if run_test(test_file):
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    
    # Return non-zero exit code if any tests failed
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 