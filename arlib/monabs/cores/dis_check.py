"""
Disjunctive over-approximation
"""
from typing import List
import z3


def compact_check_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef], res_label: List):
    f = z3.BoolVal(False)

    conditions = []
    for i in range(len(res_label)):
        if res_label[i] == 2:
            conditions.append(cnt_list[i])

    if len(conditions) == 0:
        return

    f = z3.Or(conditions)

    if z3.is_false(f):
        return

    solver = z3.Solver()
    g = z3.And(precond, f)
    solver.add(g)
    res = solver.check()
    if res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif res == z3.sat:
        m = solver.model()
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                res_label[i] = 1
    else:
        return
    compact_check_cached(precond, cnt_list, res_label)


def disjunctive_check_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    res = [2] * len(cnt_list)  # 0 means unsat, 1 means sat, 2 means "unknown"
    compact_check_cached(precond, cnt_list, res)
    return res


def compact_check_incremental_cached(solver: z3.Solver, precond: z3.ExprRef, cnt_list: List[z3.ExprRef], res_label: List[int]):
    f = z3.BoolVal(False)

    conditions = []
    for i, label in enumerate(res_label):
        if label == 2:
            conditions.append(cnt_list[i])

    if len(conditions) == 0:
        return

    f = z3.Or(conditions)

    if z3.is_false(f):
        return

    solver.push()
    solver.add(f)
    res = solver.check()
    if res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif res == z3.sat:
        m = solver.model()
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                res_label[i] = 1
    else:
        return
    solver.pop()
    compact_check_incremental_cached(solver, precond, cnt_list, res_label)


def disjunctive_check_incremental_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    results = [2] * len(cnt_list)
    solver = z3.Solver()
    solver.add(precond)
    compact_check_incremental_cached(solver, precond, cnt_list, results)
    return results