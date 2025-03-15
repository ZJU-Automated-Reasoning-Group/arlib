# Extended Mini-Symex: Symbolic Execution for Python

This is an extended version of the mini-symex framework that adds support for more Python features. The original mini-symex framework provided basic concolic execution for integers, and this extended version adds support for:

- Floating-point numbers
- Strings
- Lists
- Dictionaries
- Object-oriented programming
- Enhanced control flow (symbolic loops, exception handling)
- Improved path exploration strategies
- Automatic instrumentation

## Overview

Symbolic execution is a program analysis technique that executes programs with symbolic inputs instead of concrete values. This allows the analysis to explore multiple execution paths and find inputs that trigger specific behaviors or bugs.

Concolic execution (a portmanteau of concrete and symbolic) combines concrete execution with symbolic constraints to make symbolic execution more practical. It executes the program with concrete inputs while simultaneously collecting symbolic constraints along the execution path.

## Features

### Data Types

- **Integers**: Symbolic integers with arithmetic and bitwise operations
- **Floating-Point Numbers**: Symbolic floating-point numbers with arithmetic operations
- **Strings**: Symbolic strings with operations like concatenation, contains, startswith, endswith
- **Lists**: Symbolic lists with indexing, slicing, and modification
- **Dictionaries**: Symbolic dictionaries with key-value operations

### Control Flow

- **Conditionals**: Symbolic execution explores both branches of conditional statements
- **Loops**: Symbolic loops with symbolic bounds using `symbolic_range`
- **Exception Handling**: Symbolic-aware try-except-finally using `symbolic_try_except`

### Object-Oriented Programming

- **Classes and Objects**: Support for symbolic objects with method calls
- **Inheritance**: Support for class hierarchies
- **Polymorphism**: Support for method overriding

### Path Exploration

- **Coverage-Guided**: Prioritize paths that increase code coverage
- **Directed**: Guide exploration towards specific code locations

### Automatic Instrumentation

- **Function Instrumentation**: Automatically instrument functions for symbolic execution
- **Parameter Handling**: Automatically create symbolic variables for function parameters

## Usage

### Basic Usage

```python
from concolic import *

def test_function():
    x = mk_int("x")
    y = mk_int("y")
    
    if x > y:
        if x > 10:
            print("Path 1: x > y, x > 10")
        else:
            print("Path 2: x > y, x <= 10")
    else:
        print("Path 3: x <= y")

if __name__ == "__main__":
    concolic(test_function, debug=True)
```

### Using Different Data Types

```python
# Integer
x = mk_int("x")

# Float
f = mk_float("f")

# String
s = mk_str("s")

# List
lst = mk_list("lst", 3)  # Create a list with 3 elements

# Dictionary
d = mk_dict("d")
```

### Using Symbolic Control Flow

```python
# Symbolic range
for i in symbolic_range(n):
    # Loop body

# Symbolic try-except
result = symbolic_try_except(
    try_block,
    [(ExceptionType1, handler1),
     (ExceptionType2, handler2)],
    finally_block
)
```

### Using Object-Oriented Features

```python
# Create a symbolic object
obj = mk_object("obj", MyClass, arg1, arg2)

# Call methods
result = obj.my_method()
```

### Using Automatic Instrumentation

```python
# Instrument a function
instrumented = instrument_function(my_function)

# Execute with concolic execution
concolic_exec(my_function, arg1, arg2)
```

## Examples and Tests

The framework includes a comprehensive test suite in the `tests` directory. These tests cover various aspects of the framework, including basic functionality, advanced language features, and standard library support.

### Running Tests

To run a specific test file, use the following command:

```bash
cd tests
python test_file.py
```

For example:

```bash
cd tests
python test_extended.py
```

### Test Files

- **test_1.py, test_2.py**: Basic tests for the original mini-symex framework
- **test_advanced.py**: Tests for advanced Python language features
- **test_comprehensive.py**: Comprehensive tests covering a wide range of Python language features
- **test_stdlib.py**: Tests for standard library support
- **test_extended.py**: Tests for the extended mini-symex framework
- **test_out_of_bound.py**: Tests for out-of-bounds array access detection
- **test_overflow.py**: Tests for integer overflow detection
- **test_dbz.py**: Tests for division by zero detection

See the `tests/README.md` file for more details on the test suite.

## Implementation Details

The implementation extends the original mini-symex framework with:

1. New AST classes for additional data types
2. New concolic classes that wrap Python's built-in types
3. Support for symbolic control flow constructs
4. Integration with Z3 solver for constraint solving
5. Path exploration strategies

## Limitations

- Performance may degrade with complex constraints
- Some Python features are not fully supported (e.g., generators, decorators)
- External libraries may not work correctly with symbolic execution
- Path explosion can occur with deeply nested conditionals or loops

## Future Work

- Support for more Python features (sets, generators, etc.)
- Better handling of external libraries
- More sophisticated path exploration strategies
- Integration with test generation frameworks
- Support for multi-threading and concurrency 