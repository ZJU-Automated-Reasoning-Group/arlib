"""
Check them in parallel
"""
from typing import List
import z3
from concurrent.futures import ThreadPoolExecutor


def unary_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    """
    Solve precond \land cnt_i \in cnt_list one-by-one
    >>> import z3
    >>> x = z3.Int('x')
    >>> y = z3.Int('y')
    >>> precond = x >= 0
    >>> cnt_list = [x < -1, x <= 3, x <= 4]
    >>> unary_check_cached(precond, cnt_list)
    [0, 1, 1]
    """
    results = []
    solver = z3.Solver()

    solver.add(precond)  # Add the precondition
    for cnt in cnt_list:
        solver.push()  # Save the current state
        solver.add(cnt)  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            results.append(1)
        elif res == z3.unsat:
            results.append(0)
        else:
            results.append(2)

        solver.pop()  # Restore the state

    return results

def _check_single_constraint(precond: z3.ExprRef, cnt: z3.ExprRef) -> int:
    """Helper function to check a single constraint with precondition"""
    solver = z3.Solver()  # Create a new solver for thread safety
    solver.add(precond)
    solver.add(cnt)
    res = solver.check()
    if res == z3.sat:
        return 1
    elif res == z3.unsat:
        return 0
    return 2

def parallel_unary_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef], max_workers: int = None) -> List:
    # This function is very likely to be thread-unafe
    # Each thread should have an independent z3 context.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each constraint to a thread that checks it
        futures = [executor.submit(_check_single_constraint, precond, cnt) 
                  for cnt in cnt_list]
        # Collect results in order
        return [future.result() for future in futures]
    

