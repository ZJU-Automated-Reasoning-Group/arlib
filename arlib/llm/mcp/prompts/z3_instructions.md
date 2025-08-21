# MCP Solver - Z3 Mode

This document provides information about using MCP Solver with the Z3 SMT Solver backend.

## ⚠️ IMPORTANT: Solution Export Requirement ⚠️

All Z3 models MUST call `export_solution` to properly extract solutions. Without this call, results will not be captured, even if your solver finds a solution!

```python
# Always include this import
from arlib.mcp.z3 import export_solution

# After solving, always call export_solution with the appropriate parameters

## For standard constraint solving:
if solver.check() == sat:
    # For satisfiable solutions, provide the solver and variables
    export_solution(solver=solver, variables=variables)
else:
    # For unsatisfiable problems, explicitly set satisfiable=False
    print("No solution exists that satisfies all constraints.")
    export_solution(satisfiable=False, variables=variables)

## For property verification (theorem proving):
# When verifying a property, we look for a counterexample (negate the property)
# solver.add(Not(property_to_verify))

if solver.check() == sat:
    # Counterexample found (property doesn't hold)
    print("Property verification failed. Counterexample found.")
    export_solution(
        solver=solver,
        variables=variables,
        is_property_verification=True
    )
else:
    # No counterexample found (property holds for all cases)
    print("Property verified successfully.")
    export_solution(
        satisfiable=False,
        variables=variables,
        is_property_verification=True
    )
```

## Export Solution Parameters

The `export_solution` function supports the following parameters:

- `solver`: The Z3 solver object containing constraints and status
- `variables`: Dictionary mapping variable names to Z3 variables
- `satisfiable`: Optional boolean to explicitly set satisfiability status (useful for unsatisfiable problems)
- `objective`: Optional objective expression for optimization problems
- `is_property_verification`: Boolean flag indicating if this is a property verification problem

## Solution Output Guidelines

When outputting results:

- **Print high-level status messages only** (e.g., "Solution found!", "Property verified successfully")
- **DO NOT print detailed variable values** - export_solution automatically handles this
- **DO NOT print example outputs** like arrays, matrices, or individual values
- **Focus on what the solution means**, not raw data
- **Keep output minimal and meaningful**

Example:
```python
# GOOD - Simple status messages
if solver.check() == sat:
    print("Solution found!")
    export_solution(solver=solver, variables=vars)

# BAD - Printing detailed values
if solver.check() == sat:
    model = solver.model()
    print(f"Array: {[model.evaluate(arr[i]) for i in range(n)]}")  # Don't do this!
    print(f"x = {model.evaluate(x)}, y = {model.evaluate(y)}")     # Don't do this!
```

## Core Features

Z3 mode provides SMT (Satisfiability Modulo Theories) solving capabilities:

- **Rich type system**: Booleans, integers, reals, bitvectors, arrays, and more
- **Constraint solving**: Solve complex constraint satisfaction problems
- **Optimization**: Optimize with respect to objective functions
- **Quantifiers**: Express constraints with universal and existential quantifiers

## Best Practices for Problem Modeling

1. **Translate all constraints correctly**:
   - Consider edge cases and implicit constraints
   - Verify that each constraint's logical formulation matches the intended meaning
   - Be explicit about ranges, domains, and special case handling

2. **Structure your model clearly**:
   - Use descriptive variable names for readability
   - Group related constraints together
   - Build long code incrementally by adding smaller items
   - Comment complex constraint logic

3. **Use the correct export_solution call**:
   ```python
   # For satisfiable results
   export_solution(solver=solver, variables=variables)

   # For unsatisfiable results
   export_solution(satisfiable=False, variables=variables)
   ```

## Z3 Best Practices

### Working with Arrays and Sequences

When working with arrays or sequences in Z3, use Pythonic approaches:

```python
# GOOD: Use list comprehensions for variable creation
arr = [Int(f"arr_{i}") for i in range(8)]

# GOOD: Use loops for adding constraints
for i in range(7):
    solver.add(arr[i] <= arr[i+1])
```

### Common Error Patterns and Solutions

1. **"'bool' object has no attribute 'as_ast'"**:
   - **Problem**: Using Python boolean instead of Z3 Bool variable
   - **Solution**: Create a Z3 Bool variable and connect it to your Python result

2. **"'int' object has no attribute 'as_ast'"**:
   - **Problem**: Using Python integer instead of Z3 Int variable
   - **Solution**: Create Z3 Int variables and use constraints to set values

3. **Empty results even with sat solution**:
   - **Problem**: Missing export_solution call
   - **Solution**: Ensure export_solution is called with correct parameters

## Debugging Checklist

If your solution isn't being properly captured:

1. ✅ Did you import the export_solution function?
2. ✅ Did you call export_solution with the appropriate parameters?
3. ✅ Did you check if the solver found a solution before calling export_solution?
4. ✅ Did you collect all variables in a dictionary?
5. ✅ Are all variables in your dictionary Z3 variables (not Python primitives)?
