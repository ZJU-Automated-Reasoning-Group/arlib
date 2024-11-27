from typing import List
from pysmt.shortcuts import Solver, And, is_sat, is_unsat, TRUE
from pysmt.typing import BOOL


def unary_check(precond, cnt_list: List) -> List:
    """
    Solve precond ∧ cnt_i ∈ cnt_list one-by-one
    """
    results = []

    with Solver(name="z3") as solver:
        solver.add_assertion(precond)  # Add the precondition

        for cnt in cnt_list:
            solver.push()  # Save the current state
            solver.add_assertion(cnt)  # Add the current constraint
            res = solver.solve()

            if res:
                results.append(1)
            elif is_unsat(And(precond, cnt)):
                results.append(0)
            else:
                results.append(2)

            solver.pop()  # Restore the state

    return results


def unary_check_cached(precond, cnt_list: List) -> List:
    """
    Solve precond ∧ cnt_i ∈ cnt_list one-by-one
    """
    results = [None] * len(cnt_list)

    with Solver(name="z3") as solver:
        solver.add_assertion(precond)  # Add the precondition

        for i, cnt in enumerate(cnt_list):
            if results[i] is not None:
                continue

            solver.push()  # Save the current state
            solver.add_assertion(cnt)  # Add the current constraint
            res = solver.solve()

            if res:
                model = solver.get_model()
                results[i] = 1
                for j, other_cnt in enumerate(cnt_list):
                    if results[j] is None and model.get_value(other_cnt) == TRUE():
                        results[j] = 1
            elif is_unsat(And(precond, cnt)):
                results[i] = 0
            else:
                results[i] = 2

            solver.pop()  # Restore the state

    return results
