"""
TODO: should we perform bit-blasting, and implement this part in the bit-level?
"""

from typing import List, Tuple
from enum import Enum
import z3


class FSolverMode(Enum):
    COMPACT = 0  # compact check
    NONCOMPACT = 1  # noncompact check
    PARALLEL_CHECK = 2  # parallel check


m_fsolver_mode = FSolverMode.COMPACT


def compact_check_misc(precond: z3.ExprRef, cnt_list: List, res_label: List, models: List):
    """
    TODO: In our settings, as long as there is one unsat, we can stop
    However, this algorithm stops until "all the remaining ones are UNSAT (which can have one of more instances)"
    """
    f = z3.BoolVal(False)
    for i in range(len(res_label)):
        if res_label[i] == 2: f = z3.Or(f, cnt_list[i])
    if z3.is_false(f): return

    sol = z3.SolverFor("QF_BV")
    g = z3.And(precond, f)
    sol.add(g)
    s_res = sol.check()
    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2: res_label[i] = 0
    elif s_res == z3.sat:
        m = sol.model()
        models.append(m)  # counterexample
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], True)):
                res_label[i] = 1
    else:
        return
    compact_check_misc(precond, cnt_list, res_label, models)


class ForAllSolver(object):
    def __init__(self):
        self.solver_mode = FSolverMode.COMPACT
        self.fmls = []

    def push(self):
        return

    def pop(self):
        return

    def parallel_check(self):
        raise NotImplementedError

    def check(self, cnt_list: List) -> Tuple[List[int], List[z3.ModelRef]]:
        return self.compact_check(cnt_list)

    def non_compact_check(self, cnt_list: List) -> Tuple[List[int], List[z3.ModelRef]]:
        res_label = []
        models = []
        for _ in cnt_list: res_label.append(2)
        s = z3.SolverFor("QF_BV")
        s.add(z3.And(self.fmls))
        for cnt in cnt_list:
            res = s.check(cnt)  # check with assumption
            if res == z3.sat:
                res_label.append(1)
                models.append(s.model())
            elif res == z3.unsat:
                res_label.append(0)
                break
            # TODO: handle timeout
        return res_label, models

    def compact_check(self, cnt_list: List) -> Tuple[List[int], List[z3.ModelRef]]:
        """
        precond: And(self.fml)
        cnt_list: c1, c2, ..., c3

        """
        res_label = []
        models = []
        for _ in cnt_list: res_label.append(2)
        compact_check_misc(z3.And(self.fmls), cnt_list, res_label, models)
        return res_label, models
