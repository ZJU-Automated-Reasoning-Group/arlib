"""
Z3 solution extraction and formatting utilities.
"""

from typing import Any

import z3
# Track the last solution
_LAST_SOLUTION = None


def _extract_variable_values(model, variables: dict[str, Any]) -> dict[str, Any]:
    """Extract variable values from a Z3 model."""
    if not model or not variables:
        return {}

    result = {}
    for name, var in variables.items():
        try:
            val = model.eval(var, model_completion=True)
            if z3.is_int(val):
                result[name] = val.as_long()
            elif z3.is_real(val):
                result[name] = float(val.as_decimal(10))
            elif z3.is_bool(val):
                result[name] = z3.is_true(val)
            else:
                result[name] = str(val)
        except Exception as e:
            result[name] = f"Error: {e}"
    return result

def _extract_objective_value(model, objective):
    """Extract objective value from a Z3 model."""
    if not model or not objective:
        return None
    try:
        obj_val = model.eval(objective, model_completion=True)
        return obj_val.as_long() if z3.is_int(obj_val) else float(obj_val.as_decimal(10)) if z3.is_real(obj_val) else str(obj_val)
    except Exception as e:
        return f"Error: {e}"


def export_solution(
    solver=None,
    variables=None,
    objective=None,
    satisfiable=None,
    solution_dict=None,
    is_property_verification=False,
    property_verified=None,
) -> dict[str, Any]:
    """
    Extract and format solutions from a Z3 solver.

    Args:
        solver: Z3 Solver or Optimize object
        variables: Dictionary mapping variable names to Z3 variables
        objective: Z3 expression being optimized (optional)
        satisfiable: Explicitly override the satisfiability status (optional)
        solution_dict: Directly provide a solution dictionary (optional)
        is_property_verification: Flag for property verification problems (optional)
        property_verified: Boolean indicating if property was verified (optional)

    Returns:
        Dictionary containing the solution details
    """
    global _LAST_SOLUTION

    result = {
        "satisfiable": False,
        "values": {},
        "objective": None,
        "status": "unknown",
        "output": [],
    }

    # Use provided solution_dict if available
    if solution_dict is not None and isinstance(solution_dict, dict):
        result.update(solution_dict)
        result.setdefault("satisfiable", False)
        result.setdefault("values", {})
        result.setdefault("objective", None)
        result.setdefault("status", "unknown")
        result.setdefault("output", [])

        if "satisfiable" in solution_dict:
            result["status"] = "sat" if solution_dict["satisfiable"] else "unsat"

        _LAST_SOLUTION = result
        return result

    # Process solver if provided
    if solver is not None:
        if isinstance(solver, (z3.Solver, z3.Optimize)):
            status = solver.check()
            result["status"] = str(status)

            if status == z3.sat:
                result["satisfiable"] = True

                if variables is not None:
                    model = solver.model()
                    result["values"] = _extract_variable_values(model, variables)

                    if objective is not None and isinstance(solver, z3.Optimize):
                        result["objective"] = _extract_objective_value(model, objective)
        else:
            print(f"Warning: Unknown solver type: {type(solver)}")
    elif variables is not None:
        print("Warning: Variables provided but no solver. Only variable names will be included.")
        result["values"] = {name: None for name in variables}

    # Override satisfiability if explicitly provided
    if satisfiable is not None:
        result["satisfiable"] = bool(satisfiable)
        result["status"] = "sat" if result["satisfiable"] else "unsat"

    # Handle property verification cases
    if is_property_verification:
        if property_verified is not None:
            result["values"]["property_verified"] = bool(property_verified)

            if property_verified:
                result["output"].append("Property verified successfully.")
            else:
                result["output"].append("Property verification failed. Counterexample found.")
                result["satisfiable"] = True
                result["status"] = "sat"
        else:
            # Infer from solver result
            property_verified = result["status"] == "unsat"
            result["values"]["property_verified"] = property_verified

            if property_verified:
                result["output"].append("Property verified successfully.")
            else:
                result["output"].append("Property verification failed. Counterexample found.")
                result["satisfiable"] = True
                result["status"] = "sat"
    else:
        # For regular constraint solving
        if result["satisfiable"]:
            result["output"].append("Solution found.")
        else:
            result["output"].append("No solution exists that satisfies all constraints.")

    _LAST_SOLUTION = result
    return result
