"""
Forall Solver
"""
import logging
from enum import Enum

import z3

from arlib.utils.exceptions import ForAllSolverSuccess

from arlib.efsmt.efbv.efbv_parallel_utils import parallel_check_sat

logger = logging.getLogger(__name__)


def check_theory_consistency(fml: z3.BoolRef):
    """
    """
    logger.debug("checking candidate ")
    sol = z3.SolverFor("QF_BV")
    sol.add(fml)
    if sol.check() == z3.sat:
        m = sol.model()
        return sol.model()
    # return None
    raise ForAllSolverSuccess()
    # return ""  # empty core indicates SAT?


class FSolverMode(Enum):
    SEQUENTIAL = 0
    PARALLEL = 1  # parallel check
    EXTERNAL_PARALLEL = 2  # use external SMT solvers for parallel


class ForAllSolver(object):
    def __init__(self):
        # self.solver_mode = FSolverMode.PARALLEL
        self.solver_mode = FSolverMode.SEQUENTIAL
        self.forall_vars = []
        self.phi = None

    def push(self):
        return

    def pop(self):
        return

    def check(self, cnt_list):
        if self.solver_mode == FSolverMode.SEQUENTIAL:
            return self.sequential_check(cnt_list)
        elif self.solver_mode == FSolverMode.PARALLEL:
            return self.parallel_check(cnt_list)

    def sequential_check(self, cnt_list):
        """
        Check one-by-one
        """
        models = []
        s = z3.SolverFor("QF_BV")
        for cnt in cnt_list:
            res = s.check(cnt)  # check with assumption
            if res == z3.sat:
                models.append(s.model())
            elif res == z3.unsat:
                return []  # as list one is UNSAT
        return models

    def parallel_check(self, cnt_list):
        res = parallel_check_sat(cnt_list, 4)
        return res
        # raise NotImplementedError

    def get_blocking_fml(self, cnt_list):
        """
        """
        cex = self.check(cnt_list)
        if len(cex) == 0:
            # At least one Not(sub_phi) is UNSAT
            return z3.BoolVal(False)
        fmls = []
        for model in cex:
            sigma = [model.eval(vy, True) for vy in self.forall_vars]
            sub_phi = self.phi
            for j in range(len(self.forall_vars)):
                sub_phi = z3.simplify(z3.substitute(sub_phi, (self.forall_vars[j], sigma[j])))
            # block all CEX?
            fmls.append(sub_phi)
        return z3.simplify(z3.And(fmls))


def compact_check_misc(precond, cnt_list, res_label, models):
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
