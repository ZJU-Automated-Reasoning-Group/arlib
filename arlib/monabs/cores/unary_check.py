"""
Check one-by-one
"""
from typing import List
import z3


def unary_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    results = [None] * len(cnt_list)
    
    for i, cnt in enumerate(cnt_list):
        solver = z3.Solver()
        solver.add(precond)  # Add the precondition
        solver.add(cnt)  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            results[i] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2
            
    return results


def unary_check_incremental(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    results = [None] * len(cnt_list)
    solver = z3.Solver()
    
    solver.add(precond)  # Add the precondition
    for i, cnt in enumerate(cnt_list):
        solver.push()  # Save the current state
        
        solver.add(cnt)  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            results[i] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2

        solver.pop()  # Restore the state

    return results


def unary_check_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    results = [None] * len(cnt_list)

    for i, cnt in enumerate(cnt_list):
        if results[i] is not None:
            continue

        solver = z3.Solver()
        solver.add(precond)  # Add the precondition
        solver.add(cnt)  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            results[i] = 1
            for j, other_cnt in enumerate(cnt_list):
                if results[j] is None and z3.is_true(model.eval(other_cnt, model_completion=True)):
                    results[j] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2

    return results


def unary_check_incremental_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    results = [None] * len(cnt_list)
    solver = z3.Solver()

    solver.add(precond)  # Add the precondition

    for i, cnt in enumerate(cnt_list):
        if results[i] is not None:
            continue

        solver.push()  # Save the current state
        
        solver.add(cnt)  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            results[i] = 1
            for j, other_cnt in enumerate(cnt_list):
                if results[j] is None and z3.is_true(model.eval(other_cnt, model_completion=True)):
                    results[j] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2

        solver.pop()  # Restore the state

    return results