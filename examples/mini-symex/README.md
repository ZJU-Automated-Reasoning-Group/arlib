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

### State and Path Control

The mini-symex framework includes controls for limiting the number of states and paths explored during concolic execution. This is useful for:

1. Limiting resource usage for large or complex programs
2. Focusing exploration on specific parts of the program
3. Benchmarking and performance testing
4. Debugging symbolic execution issues

#### New Parameters

The `concolic()` function accepts two new parameters:

- `max_states`: Maximum number of states to explore (None for unlimited)
- `max_paths`: Maximum number of paths to explore (None for unlimited)

#### Return Value

The `concolic()` function now returns a tuple of `(crashes, stats)` where:

- `crashes`: List of inputs that lead to crashes/exceptions
- `stats`: Dictionary containing exploration statistics:
  - `states_explored`: Number of states explored
  - `paths_explored`: Number of paths explored
  - `max_queue_size`: Maximum size of the exploration queue
  - `solver_calls`: Number of calls to the Z3 solver
  - `solver_sat`: Number of satisfiable results from the solver
  - `solver_unsat`: Number of unsatisfiable results from the solver

#### Example Usage

```python
from concolic import *

def test_function():
    x = mk_int("x")
    y = mk_int("y")
    
    if x > 0:
        if y > 0:
            print(f"Path 1: x={x}, y={y}")
        else:
            print(f"Path 2: x={x}, y={y}")
    else:
        print(f"Path 3: x={x}, y={y}")

# Run with unlimited states and paths
crashes, stats = concolic(test_function, debug=True)
print(f"Explored {stats['states_explored']} states and {stats['paths_explored']} paths")

# Run with a maximum of 5 states
crashes, stats = concolic(test_function, debug=True, max_states=5)
print(f"Explored {stats['states_explored']} states and {stats['paths_explored']} paths")

# Run with a maximum of 3 paths
crashes, stats = concolic(test_function, debug=True, max_paths=3)
print(f"Explored {stats['states_explored']} states and {stats['paths_explored']} paths")
```

#### Advanced Usage

The state and path controls can also be used with the other concolic execution functions:

```python
# Coverage-guided exploration with state and path limits
crashes, stats = concolic_coverage_guided(test_function, max_states=10, max_paths=20)

# Directed exploration with state and path limits
crashes, stats = concolic_directed(test_function, target_line=42, max_states=10, max_paths=20)

# Automatic instrumentation with state and path limits
crashes, stats = concolic_exec(some_function, arg1, arg2, max_states=10, max_paths=20)
```

#### Performance Considerations

- Setting appropriate limits can significantly reduce execution time for complex programs
- For most programs, the number of paths grows exponentially with program size
- Setting `max_states` is generally more effective than `max_paths` for limiting resource usage
- The statistics returned can be used to tune the limits for optimal performance

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