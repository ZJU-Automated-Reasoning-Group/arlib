# Mini-Symex Test Suite

This directory contains test cases for the mini-symex symbolic execution framework. The tests cover various aspects of the framework, including basic functionality, advanced language features, and standard library support.

## Test Files

- **test_1.py, test_2.py**: Basic tests for the original mini-symex framework
- **test_advanced.py**: Tests for advanced Python language features
- **test_base64.py**: Tests for base64 encoding/decoding
- **test_comprehensive.py**: Comprehensive tests covering a wide range of Python language features
- **test_dbz.py**: Tests for division by zero detection
- **test_extended.py**: Tests for the extended mini-symex framework
- **test_out_of_bound.py**: Tests for out-of-bounds array access detection
- **test_overflow.py**: Tests for integer overflow detection
- **test_stdlib.py**: Tests for standard library support
- **test_symbolic.py**: Tests for symbolic execution core functionality

## Running the Tests

To run a specific test file, use the following command from this directory:

```bash
python test_file.py
```

For example:

```bash
python test_extended.py
```

All test files use the `concolic()` function to run the tests with symbolic execution. The tests will explore multiple execution paths and report any errors or exceptions encountered.

## Test Organization

The tests are organized to cover different aspects of the symbolic execution framework:

1. **Basic Data Types and Operations**: Tests for integers, floats, strings, lists, and dictionaries
2. **Control Flow**: Tests for conditional statements, loops, and exception handling
3. **Object-Oriented Features**: Tests for classes, objects, inheritance, and polymorphism
4. **Advanced Language Features**: Tests for decorators, metaclasses, context managers, generators, etc.
5. **Standard Library Support**: Tests for various modules from Python's standard library

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new file with a descriptive name (e.g., `test_feature.py`)
2. Import the necessary modules from the parent directory:
   ```python
   import sys
   import os
   
   # Add parent directory to path to allow importing modules
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   
   from concolic import *
   ```
3. Define test functions with descriptive names (e.g., `test_feature_name`)
4. Use the `concolic()` function in the main block to run the tests:
   ```python
   if __name__ == "__main__":
       print("\n=== Running test_feature_name ===")
       concolic(test_feature_name, debug=True, exit_on_err=False)
   ```

## Troubleshooting

If you encounter issues running the tests:

1. Make sure the parent directory is in the Python path (using `sys.path.append` as shown above)
2. Check that you're using the `concolic()` function to run the tests
3. Verify that symbolic variables are created using the appropriate functions (`mk_int`, `mk_float`, etc.)
4. Ensure that path constraints are properly added using `add_pc()` 