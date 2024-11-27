from typing import List
from pysmt.shortcuts import Symbol, Or, And, Solver, Not, TRUE, is_unsat, FALSE
from pysmt.typing import BOOL


def compact_check_misc(precond, cnt_list: List, res_label: List[int]):
    f = FALSE()
    for i in range(len(res_label)):
        if res_label[i] == 2:
            f = Or(f, cnt_list[i])
    if is_unsat(f):
        return

    with Solver(name="z3") as sol:
        g = And(precond, f)
        sol.add_assertion(g)
        s_res = sol.solve()

        if not s_res:
            for i in range(len(res_label)):
                if res_label[i] == 2:
                    res_label[i] = 0
        elif s_res:
            m = sol.get_model()
            for i in range(len(res_label)):
                if res_label[i] == 2 and m.get_value(cnt_list[i]) == TRUE():
                    res_label[i] = 1
        else:
            return
    compact_check_misc(precond, cnt_list, res_label)


def disjunctive_check(precond, cnt_list: List) -> List[int]:
    res = [2] * len(cnt_list)  # 0 means unsat, 1 means sat, 2 means "unknown"
    compact_check_misc(precond, cnt_list, res)
    return res


def compact_check_misc_incremental(solver, precond, cnt_list: List, res_label: List[int]):
    f = FALSE()
    for i, label in enumerate(res_label):
        if label == 2:
            f = Or(f, cnt_list[i])

    if is_unsat(f):
        return

    solver.push()
    solver.add_assertion(f)
    s_res = solver.solve()

    if not s_res:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif s_res:
        m = solver.get_model()
        for i in range(len(res_label)):
            if res_label[i] == 2 and m.get_value(cnt_list[i]) == TRUE():
                res_label[i] = 1

    solver.pop()
    compact_check_misc_incremental(solver, precond, cnt_list, res_label)


def disjunctive_check_incremental(precond, cnt_list: List) -> List[int]:
    res = [2] * len(cnt_list)
    with Solver(name="z3") as solver:
        solver.add_assertion(precond)
        compact_check_misc_incremental(solver, precond, cnt_list, res)
    return res
