#!/usr/bin/env python3
"""
Test cases for Python's standard library features using the extended mini-symex framework.
This file tests interaction with various modules from Python's standard library.
"""



import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
import re
import json
import datetime
import math
import random
import collections
import itertools
# Add parent directory to path to allow importing modules
###########################################
# 1. Math and Numeric Operations
###########################################

def test_math_module():
    """Test math module functions with symbolic values"""
    print("\n=== Testing Math Module ===")
    
    x = mk_float("x")
    
    # Basic math functions
    if math.sin(x) > 0.5:
        print(f"math.sin({x}) > 0.5")
    
    if math.cos(x) < 0.5:
        print(f"math.cos({x}) < 0.5")
    
    if math.sqrt(x) > 3:
        print(f"math.sqrt({x}) > 3, x={x}")
    
    # Constants
    if x > math.pi:
        print(f"{x} > π")
    
    if x < math.e:
        print(f"{x} < e")

def test_random_module():
    """Test random module with symbolic seeds"""
    print("\n=== Testing Random Module ===")
    
    seed = mk_int("seed")
    
    # Set the seed for reproducibility
    random.seed(seed)
    
    # Generate random numbers
    r1 = random.random()  # Float between 0 and 1
    r2 = random.randint(1, 100)  # Integer between 1 and 100
    
    print(f"With seed {seed}: random.random()={r1}, random.randint(1,100)={r2}")

###########################################
# 2. String Processing
###########################################

def test_string_methods():
    """Test string methods from the standard library"""
    print("\n=== Testing String Methods ===")
    
    s = mk_str("s")
    
    # String methods
    if s.upper() == "PYTHON":
        print(f"s.upper() == 'PYTHON': s='{s}'")
    
    if s.lower() == "python":
        print(f"s.lower() == 'python': s='{s}'")
    
    if s.strip() == "python":
        print(f"s.strip() == 'python': s='{s}'")
    
    # String splitting and joining
    parts = s.split()
    if len(parts) > 2:
        print(f"s.split() has more than 2 parts: s='{s}', parts={parts}")
    
    joined = "-".join(parts)
    if len(joined) > len(s):
        print(f"'-'.join(parts) is longer than s: s='{s}', joined='{joined}'")

def test_regular_expressions():
    """Test regular expressions with symbolic strings"""
    print("\n=== Testing Regular Expressions ===")
    
    s = mk_str("s")
    
    # Match patterns
    if re.match(r"^py", s.lower()):
        print(f"s starts with 'py': s='{s}'")
    
    if re.search(r"th.n$", s.lower()):
        print(f"s ends with 'th?n': s='{s}'")
    
    # Replace patterns
    replaced = re.sub(r"[aeiou]", "*", s.lower())
    if len(replaced) == len(s):
        print(f"Vowels replaced: original='{s}', replaced='{replaced}'")

###########################################
# 3. Data Structures
###########################################

def test_collections_module():
    """Test collections module with symbolic values"""
    print("\n=== Testing Collections Module ===")
    
    # Create a Counter
    lst = mk_list("lst", 5)
    counter = collections.Counter(lst)
    
    # Check most common elements
    most_common = counter.most_common(1)
    if most_common and most_common[0][1] > 2:
        print(f"Most common element appears more than twice: {most_common}")
    
    # Create a defaultdict
    d = collections.defaultdict(int)
    
    x = mk_int("x")
    y = mk_int("y")
    
    d["x"] = x
    d["y"] = y
    
    # Access a non-existent key (will create it with default value 0)
    z = d["z"]
    
    if x + y + z > 10:
        print(f"x + y + z > 10: x={x}, y={y}, z={z}")

def test_json_module():
    """Test JSON serialization and deserialization"""
    print("\n=== Testing JSON Module ===")
    
    # Create a dictionary with symbolic values
    x = mk_int("x")
    s = mk_str("s")
    
    data = {
        "number": x,
        "string": s,
        "list": [1, 2, x],
        "nested": {"key": s}
    }
    
    # Serialize to JSON
    json_str = json.dumps(data)
    print(f"JSON string: {json_str}")
    
    # Deserialize from JSON
    parsed = json.loads(json_str)
    
    if parsed["number"] > 10:
        print(f"Parsed number > 10: {parsed['number']}")
    
    if len(parsed["string"]) > 5:
        print(f"Parsed string length > 5: '{parsed['string']}'")

###########################################
# 4. Date and Time
###########################################

def test_datetime_module():
    """Test datetime module with symbolic values"""
    print("\n=== Testing Datetime Module ===")
    
    # Create symbolic values for date components
    year = mk_int("year")
    month = mk_int("month")
    day = mk_int("day")
    
    # Ensure valid date ranges
    if 2000 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 28:
        try:
            # Create a date object
            date = datetime.date(year, month, day)
            
            # Check if it's a future date
            today = datetime.date.today()
            if date > today:
                print(f"Future date: {date}, today: {today}")
            else:
                print(f"Past or present date: {date}, today: {today}")
            
            # Calculate time delta
            delta = date - today
            if delta.days > 365:
                print(f"More than a year in the future: {delta.days} days")
            
        except ValueError as e:
            print(f"Invalid date: {year}-{month}-{day}, error: {e}")

###########################################
# 5. File System Operations
###########################################

def test_os_module():
    """Test os module with symbolic values"""
    print("\n=== Testing OS Module ===")
    
    # Get current directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    
    # List directory contents
    files = os.listdir(".")
    print(f"Files in current directory: {files[:5]}...")
    
    # Check file existence with symbolic path components
    base_dir = "."
    subdir = mk_str("subdir")
    
    # Ensure subdir is a valid directory name (no slashes, dots)
    if re.match(r"^[a-zA-Z0-9_-]+$", subdir):
        path = os.path.join(base_dir, subdir)
        exists = os.path.exists(path)
        print(f"Checking if path exists: {path}, result: {exists}")

def test_path_manipulation():
    """Test path manipulation with symbolic values"""
    print("\n=== Testing Path Manipulation ===")
    
    base = mk_str("base")
    ext = mk_str("ext")
    
    # Ensure valid filename components
    if re.match(r"^[a-zA-Z0-9_-]+$", base) and re.match(r"^[a-zA-Z0-9]+$", ext):
        # Create a filename
        filename = f"{base}.{ext}"
        
        # Split the path
        dirname, basename = os.path.split(filename)
        print(f"Split path: dirname='{dirname}', basename='{basename}'")
        
        # Split extension
        name, extension = os.path.splitext(filename)
        print(f"Split extension: name='{name}', extension='{extension}'")
        
        # Join paths
        full_path = os.path.join("/tmp", filename)
        print(f"Joined path: {full_path}")

###########################################
# 6. Advanced Standard Library Features
###########################################

def test_functional_tools():
    """Test functional programming tools"""
    print("\n=== Testing Functional Tools ===")
    
    x = mk_int("x")
    y = mk_int("y")
    
    # Create a list of numbers
    numbers = [x, y, x+y, x*y]
    
    # Use map to apply a function to each element
    squared = list(map(lambda n: n*n, numbers))
    
    if sum(squared) > 100:
        print(f"Sum of squared numbers > 100: numbers={numbers}, squared={squared}")
    
    # Use filter to select elements
    filtered = list(filter(lambda n: n > 10, numbers))
    
    if len(filtered) >= 2:
        print(f"At least 2 numbers > 10: numbers={numbers}, filtered={filtered}")

def test_itertools():
    """Test itertools module with symbolic values"""
    print("\n=== Testing Itertools Module ===")
    
        
    x = mk_int("x")
    y = mk_int("y")
    
    # Create lists with symbolic values
    list1 = [1, x, 3]
    list2 = [y, 5, 6]
    
    # Product
    product = list(itertools.product(list1, list2))
    print(f"Product of {list1} and {list2}: {product}")
    
    # Combinations
    combinations = list(itertools.combinations(list1, 2))
    print(f"Combinations of {list1}: {combinations}")
    
    # Chain
    chained = list(itertools.chain(list1, list2))
    if sum(chained) > 20:
        print(f"Sum of chained lists > 20: {chained}, sum={sum(chained)}")

###########################################
# Main Test Runner
###########################################

if __name__ == "__main__":
    print("=== Running Standard Library Tests ===")
    
    # Math and numeric operations
    concolic(test_math_module, debug=True, exit_on_err=False)
    concolic(test_random_module, debug=True, exit_on_err=False)
    
    # String processing
    concolic(test_string_methods, debug=True, exit_on_err=False)
    concolic(test_regular_expressions, debug=True, exit_on_err=False)
    
    # Data structures
    concolic(test_collections_module, debug=True, exit_on_err=False)
    concolic(test_json_module, debug=True, exit_on_err=False)
    
    # Date and time
    concolic(test_datetime_module, debug=True, exit_on_err=False)
    
    # File system operations
    concolic(test_os_module, debug=True, exit_on_err=False)
    concolic(test_path_manipulation, debug=True, exit_on_err=False)
    
    # Advanced standard library features
    concolic(test_functional_tools, debug=True, exit_on_err=False)
    concolic(test_itertools, debug=True, exit_on_err=False)
    
    print("\n=== All Standard Library Tests Completed ===") 