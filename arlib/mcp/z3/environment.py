"""
Z3 environment module for secure execution of Z3 code.
"""

import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any

import z3
# Global variable to store the solution
_LAST_SOLUTION = None


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds: float):
    """Context manager for limiting execution time of code blocks."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_z3_code(code_string: str, timeout: float = 4.0) -> dict[str, Any]:
    """
    Execute Z3 Python code in a secure environment with timeout handling.

    Args:
        code_string: The Z3 Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with execution results
    """
    from .solution import export_solution
    from . import templates

    global _LAST_SOLUTION
    _LAST_SOLUTION = None

    # Pre-process the code to remove imports and add global declarations
    code_lines = code_string.split("\n")
    processed_code = []

    for line in code_lines:
        # Skip Z3 import lines
        if (line.strip().startswith("from z3 import") or
            line.strip().startswith("import z3")):
            continue
        else:
            processed_code.append(line)

    processed_code_string = "\n".join(processed_code)

    # Add preamble for global solver variables
    processed_code_string = (
        """
# Make solver variables globally accessible
global solver, s, z3_solver
solver = None
s = None
z3_solver = None

"""
        + processed_code_string
    )

    # Create restricted globals with Z3 symbols
    restricted_globals = {
        # Basic Python builtins
        "Exception": Exception,
        "ImportError": ImportError,
        "NameError": NameError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        # Z3 module and symbols
        "z3": z3,
        "export_solution": export_solution,
        # Global solver variables
        "solver": None,
        "s": None,
        "z3_solver": None,
        # Template modules
        "templates": templates,
    }

    # Add Z3 symbols to simulate 'from z3 import *'
    for name in dir(z3):
        if not name.startswith("_") and name not in restricted_globals:
            restricted_globals[name] = getattr(z3, name)

    # Add template functions
    restricted_globals.update({
        "constraint_satisfaction_template": templates.function_templates.constraint_satisfaction_template,
        "optimization_template": templates.function_templates.optimization_template,
        "array_template": templates.function_templates.array_template,
        "quantifier_template": templates.function_templates.quantifier_template,
        "demo_template": templates.function_templates.demo_template,
        "smallest_subset_with_property": templates.subset_templates.smallest_subset_with_property,
        "subset_selection_template": templates.subset_templates.subset_selection_template,
        "array_is_sorted": templates.z3_templates.array_is_sorted,
        "all_distinct": templates.z3_templates.all_distinct,
        "array_contains": templates.z3_templates.array_contains,
        "exactly_k": templates.z3_templates.exactly_k,
        "at_most_k": templates.z3_templates.at_most_k,
        "at_least_k": templates.z3_templates.at_least_k,
        "function_is_injective": templates.z3_templates.function_is_injective,
        "function_is_surjective": templates.z3_templates.function_is_surjective,
    })

    result = {
        "status": "unknown",
        "error": None,
        "output": [],
        "solution": None,
        "execution_time": 0,
    }

    # Capture print output
    original_stdout = sys.stdout
    from io import StringIO
    captured_output = StringIO()
    sys.stdout = captured_output

    start_time = time.time()

    try:
        with time_limit(timeout):
            local_vars = {}
            exec(processed_code_string, restricted_globals, local_vars)

            # Check for solution
            if "solution" in local_vars:
                result["solution"] = local_vars["solution"]
                result["status"] = "success"
            elif _LAST_SOLUTION is not None:
                result["solution"] = _LAST_SOLUTION
                result["status"] = "success"
            else:
                from .solution import _LAST_SOLUTION as solution_last_solution
                if solution_last_solution is not None:
                    result["solution"] = solution_last_solution
                    result["status"] = "success"
                else:
                    result["status"] = "no_solution"
                    result["error"] = "No solution was exported. Make sure to call export_solution()"

    except TimeoutException as e:
        result["status"] = "timeout"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e!s}"
        result["traceback"] = traceback.format_exc()
    finally:
        sys.stdout = original_stdout
        result["execution_time"] = time.time() - start_time
        result["output"] = (
            captured_output.getvalue().strip().split("\n")
            if captured_output.getvalue()
            else []
        )

    return result
