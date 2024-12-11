"""
Disjunctive over-approximation
"""
from typing import List
import z3


def compact_check_misc(precond: z3.ExprRef, cnt_list: [z3.ExprRef], res_label: List):
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

    # sol = z3.SolverFor("QF_BV")
    sol = z3.Solver()
    g = z3.And(precond, f)
    sol.add(g)
    s_res = sol.check()
    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif s_res == z3.sat:
        m = sol.model()
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], True)):
                res_label[i] = 1
    else:
        return
    compact_check_misc(precond, cnt_list, res_label)


def disjunctive_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    """
    Given a precond G and a set of cnts: f1, f2,..., fn
    Decis if the following cnts are satisfiable:
           And(G, f1), And(G, f2), ..., And(G, fn)
    Examples:
    >>> from z3 import *
    >>> x, y = Reals('x y')
    >>> pre = x > 100
    >>> f1 = x > 0; f2 = And(x > y, x < y); f3 = x < 3
    >>> cnts = [f1, f2, f3]
    >>> assert disjunctive_check(pre, cnts) == [1, 0, 0]
    """
    res = [2] * len(cnt_list)  # 0 means unsat, 1 means at, 2 means "unknown"
    compact_check_misc(precond, cnt_list, res)
    return res


def compact_check_misc_incremental(solver: z3.Solver, precond: z3.ExprRef, cnt_list: List[z3.ExprRef], res_label: List[int]):
    f = z3.BoolVal(False)
    
    conditions = []
    for i, label in enumerate(res_label):
        if label == 2:
            conditions.append(cnt_list[i])

    if len(conditions) == 0:
        return  # 如果没有需要检查的条件，直接返回

    f = z3.Or(conditions)

    if z3.is_false(f):
        return

    solver.push()
    solver.add(f)
    s_res = solver.check()

    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif s_res == z3.sat:
        m = solver.model()
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                res_label[i] = 1

    solver.pop()
    compact_check_misc_incremental(solver, precond, cnt_list, res_label)


def disjunctive_check_incremental(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    """
    Given a precondition G and a set of constraints: f1, f2, ..., fn
    Decide if the following constraints are satisfiable:
           And(G, f1), And(G, f2), ..., And(G, fn)
    """
    res = [2] * len(cnt_list)
    solver = z3.Solver()
    solver.add(precond)
    compact_check_misc_incremental(solver, precond, cnt_list, res)
    return res
