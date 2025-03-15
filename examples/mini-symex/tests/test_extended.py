#!/usr/bin/env python3
"""
Test cases for the extended mini-symex framework.
This file tests the new features added to the framework.
"""


import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules

###########################################
# 1. Float Operations
###########################################

def test_float_operations():
    """Test float operations with symbolic values"""
    print("\n=== Testing Float Operations ===")
    
    # Create symbolic float variables
    x = mk_float("x")
    y = mk_float("y")
    
    # Test addition
    z = x + y
    if z > 10.0:
        print(f"Path 1: {x} + {y} = {z} > 10.0")
    else:
        print(f"Path 2: {x} + {y} = {z} <= 10.0")
    
    # Test subtraction
    z = x - y
    if z < 0.0:
        print(f"Path 3: {x} - {y} = {z} < 0.0")
    else:
        print(f"Path 4: {x} - {y} = {z} >= 0.0")
    
    # Test multiplication
    z = x * y
    if z > 5.0:
        print(f"Path 5: {x} * {y} = {z} > 5.0")
    else:
        print(f"Path 6: {x} * {y} = {z} <= 5.0")
    
    # Test division
    if y != 0.0:
        z = x / y
        if z > 2.0:
            print(f"Path 7: {x} / {y} = {z} > 2.0")
        else:
            print(f"Path 8: {x} / {y} = {z} <= 2.0")
    else:
        print(f"Path 9: Division by zero avoided")

###########################################
# 2. String Operations
###########################################

def test_string_operations():
    """Test string operations with symbolic values"""
    print("\n=== Testing String Operations ===")
    
    # Create symbolic string variable
    s = mk_str("s")
    
    # Test string length
    if len(s) > 5:
        print(f"Path 1: Length of '{s}' is > 5")
    else:
        print(f"Path 2: Length of '{s}' is <= 5")
    
    # Test string startswith
    if s.startswith("hello"):
        print(f"Path 3: '{s}' starts with 'hello'")
    else:
        print(f"Path 4: '{s}' does not start with 'hello'")
    
    # Test string concatenation
    t = s + " world"
    if len(t) > 10:
        print(f"Path 5: Length of '{t}' is > 10")
    else:
        print(f"Path 6: Length of '{t}' is <= 10")

###########################################
# 3. List Operations
###########################################

def test_list_operations():
    """Test list operations with symbolic values"""
    print("\n=== Testing List Operations ===")
    
    # Create symbolic list variable
    lst = mk_list("lst", 3)
    
    # Test list indexing
    if lst[0] > 5:
        print(f"Path 1: lst[0] = {lst[0]} > 5")
    else:
        print(f"Path 2: lst[0] = {lst[0]} <= 5")
    
    # Test list modification
    lst[1] = 10
    if lst[1] == 10:
        print(f"Path 3: lst[1] = {lst[1]} == 10")
    else:
        print(f"Path 4: lst[1] = {lst[1]} != 10")
    
    # Test list append
    lst.append(20)
    if len(lst) == 4:
        print(f"Path 5: Length of lst after append is {len(lst)} == 4")
    else:
        print(f"Path 6: Length of lst after append is {len(lst)} != 4")

###########################################
# 4. Dictionary Operations
###########################################

def test_dict_operations():
    """Test dictionary operations with symbolic values"""
    print("\n=== Testing Dictionary Operations ===")
    
    # Create symbolic dictionary variable
    d = mk_dict("d")
    
    # Test dictionary key presence
    if "key1" in d:
        print(f"Path 1: 'key1' is in d with value {d['key1']}")
    else:
        print(f"Path 2: 'key1' is not in d")
    
    # Test dictionary modification
    d["key2"] = 20
    if d["key2"] == 20:
        print(f"Path 3: d['key2'] = {d['key2']} == 20")
    else:
        print(f"Path 4: d['key2'] = {d['key2']} != 20")
    
    # Test dictionary get method
    value = d.get("key3", 30)
    if value == 30:
        print(f"Path 5: d.get('key3', 30) = {value} == 30")
    else:
        print(f"Path 6: d.get('key3', 30) = {value} != 30")

###########################################
# 5. Exception Handling
###########################################

def test_exception_handling():
    """Test exception handling with symbolic values"""
    print("\n=== Testing Exception Handling ===")
    
    # Create symbolic variables
    x = mk_int("x")
    y = mk_int("y")
    
    try:
        # Division that might cause ZeroDivisionError
        result = x / y
        print(f"Path 1: {x} / {y} = {result}")
    except ZeroDivisionError:
        print(f"Path 2: Caught ZeroDivisionError for {x} / {y}")
    finally:
        print(f"Path 3: Finally block executed")
    
    try:
        # List access that might cause IndexError
        lst = [1, 2, 3]
        value = lst[x]
        print(f"Path 4: lst[{x}] = {value}")
    except IndexError:
        print(f"Path 5: Caught IndexError for lst[{x}]")

###########################################
# 6. Object-Oriented Features
###########################################

class Counter:
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    def increment(self, amount=1):
        self.value += amount
        return self.value
    
    def decrement(self, amount=1):
        self.value -= amount
        return self.value

def test_object_oriented():
    """Test object-oriented features with symbolic values"""
    print("\n=== Testing Object-Oriented Features ===")
    
    # Create symbolic variable
    x = mk_int("x")
    
    # Create object with symbolic value
    counter = Counter(x)
    
    # Test method calls
    result = counter.increment(5)
    if result > 10:
        print(f"Path 1: counter.increment(5) = {result} > 10")
    else:
        print(f"Path 2: counter.increment(5) = {result} <= 10")
    
    result = counter.decrement(3)
    if result < 0:
        print(f"Path 3: counter.decrement(3) = {result} < 0")
    else:
        print(f"Path 4: counter.decrement(3) = {result} >= 0")

###########################################
# 7. Symbolic Range
###########################################

def test_symbolic_range():
    """Test symbolic range with symbolic values"""
    print("\n=== Testing Symbolic Range ===")
    
    # Create symbolic variable
    n = mk_int("n")
    
    # Test symbolic range
    sum_val = 0
    for i in symbolic_range(1, n + 1):
        sum_val += i
    
    if sum_val > 10:
        print(f"Path 1: Sum from 1 to {n} is {sum_val} > 10")
    else:
        print(f"Path 2: Sum from 1 to {n} is {sum_val} <= 10")

###########################################
# 8. Automatic Instrumentation
###########################################

def add(a, b):
    """Simple function to add two numbers"""
    return a + b

def test_instrumentation():
    """Test automatic instrumentation of functions"""
    print("\n=== Testing Automatic Instrumentation ===")
    
    # Create instrumented version of the function
    instrumented_add = instrument_function(add)
    
    # Call with concrete values
    result = instrumented_add(5, 7)
    print(f"Path 1: instrumented_add(5, 7) = {result}")
    
    # Call with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    result = instrumented_add(x, y)
    
    if result > 10:
        print(f"Path 2: instrumented_add({x}, {y}) = {result} > 10")
    else:
        print(f"Path 3: instrumented_add({x}, {y}) = {result} <= 10")

###########################################
# 9. Coverage-Guided Execution
###########################################

def complex_function(a, b, c):
    """Function with multiple paths for coverage testing"""
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            else:
                return a + b - c
        else:
            if c > 0:
                return a - b + c
            else:
                return a - b - c
    else:
        if b > 0:
            if c > 0:
                return -a + b + c
            else:
                return -a + b - c
        else:
            if c > 0:
                return -a - b + c
            else:
                return -a - b - c

def test_coverage_guided():
    """Test coverage-guided execution"""
    print("\n=== Testing Coverage-Guided Execution ===")
    
    # Create symbolic variables
    a = mk_int("a")
    b = mk_int("b")
    c = mk_int("c")
    
    # Call function with symbolic values
    result = complex_function(a, b, c)
    
    if result > 0:
        print(f"Path 1: complex_function({a}, {b}, {c}) = {result} > 0")
    else:
        print(f"Path 2: complex_function({a}, {b}, {c}) = {result} <= 0")

# Run all tests
if __name__ == "__main__":
    print("\n=== Testing Float Operations ===")
    concolic(test_float_operations, debug=True, exit_on_err=False)
    
    print("\n=== Testing String Operations ===")
    concolic(test_string_operations, debug=True, exit_on_err=False)
    
    print("\n=== Testing List Operations ===")
    concolic(test_list_operations, debug=True, exit_on_err=False)
    
    print("\n=== Testing Dictionary Operations ===")
    concolic(test_dict_operations, debug=True, exit_on_err=False)
    
    print("\n=== Testing Exception Handling ===")
    concolic(test_exception_handling, debug=True, exit_on_err=False)
    
    print("\n=== Testing Object-Oriented Features ===")
    concolic(test_object_oriented, debug=True, exit_on_err=False)
    
    print("\n=== Testing Symbolic Range ===")
    concolic(test_symbolic_range, debug=True, exit_on_err=False)
    
    print("\n=== Testing Automatic Instrumentation ===")
    concolic(test_instrumentation, debug=True, exit_on_err=False)
    
    print("\n=== Testing Coverage-Guided Execution ===")
    concolic_coverage_guided(test_coverage_guided) 