# State and Path Control in Mini-Symex

This document explains the state and path control features added to the mini-symex framework.

## Overview

The mini-symex framework now includes controls for limiting the number of states and paths explored during concolic execution. This is useful for:

1. Limiting resource usage for large or complex programs
2. Focusing exploration on specific parts of the program
3. Benchmarking and performance testing
4. Debugging symbolic execution issues

## New Parameters

The `concolic()` function now accepts two new parameters:

- `max_states`: Maximum number of states to explore (None for unlimited)
- `max_paths`: Maximum number of paths to explore (None for unlimited)

## Return Value

The `concolic()` function now returns a tuple of `(crashes, stats)` where:

- `crashes`: List of inputs that lead to crashes/exceptions
- `stats`: Dictionary containing exploration statistics:
  - `states_explored`: Number of states explored
  - `paths_explored`: Number of paths explored
  - `max_queue_size`: Maximum size of the exploration queue
  - `solver_calls`: Number of calls to the Z3 solver
  - `solver_sat`: Number of satisfiable results from the solver
  - `solver_unsat`: Number of unsatisfiable results from the solver

## Example Usage

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

## Advanced Usage

The state and path controls can also be used with the other concolic execution functions:

```python
# Coverage-guided exploration with state and path limits
crashes, stats = concolic_coverage_guided(test_function, max_states=10, max_paths=20)

# Directed exploration with state and path limits
crashes, stats = concolic_directed(test_function, target_line=42, max_states=10, max_paths=20)

# Automatic instrumentation with state and path limits
crashes, stats = concolic_exec(some_function, arg1, arg2, max_states=10, max_paths=20)
```

## Performance Considerations

- Setting appropriate limits can significantly reduce execution time for complex programs
- For most programs, the number of paths grows exponentially with program size
- Setting `max_states` is generally more effective than `max_paths` for limiting resource usage
- The statistics returned can be used to tune the limits for optimal performance 