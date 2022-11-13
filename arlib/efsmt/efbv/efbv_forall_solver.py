"""
Forall Solver
"""
import logging
from typing import List

import z3

# from arlib.utils.exceptions import ForAllSolverSuccess
from arlib.efsmt.efbv.efbv_forall_solver_helper import parallel_check_candidates
from arlib.efsmt.efbv.efbv_utils import FSolverMode

logger = logging.getLogger(__name__)


m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD


class ForAllSolver(object):
    def __init__(self, ctx: z3.Context):
        # self.forall_vars = []
        self.ctx = ctx  # the Z3 context of the main thread
        # self.phi = None

    def push(self):
        return

    def pop(self):
        return

    def check(self, cnt_list: List[z3.BoolRef]):
        if m_forall_solver_strategy == FSolverMode.SEQUENTIAL:
            return self.sequential_check(cnt_list)
        elif m_forall_solver_strategy == FSolverMode.PARALLEL_THREAD:
            return self.parallel_check_thread(cnt_list)
        elif m_forall_solver_strategy == FSolverMode.PARALLEL_PROCESS:
            return self.parallel_check_process(cnt_list)
        else:
            raise NotImplementedError

    def sequential_check(self, cnt_list: List[z3.BoolRef]):
        """
        Check one-by-one
        """
        models = []
        for cnt in cnt_list:
            s = z3.SolverFor("QF_BV")
            s.add(cnt)
            res = s.check()
            if res == z3.sat:
                m = s.model()
                models.append(s.model())
            elif res == z3.unsat:
                return []  # at least one is UNSAT
        return models

    def parallel_check_thread(self, cnt_list: List[z3.BoolRef]):
        """
        Solve each formula in cnt_list in parallel
        """
        logger.debug("Forall solver: Parallel checking the candidates")
        models_in_other_ctx = parallel_check_candidates(cnt_list, num_workers=4)
        res = []  # translate the model to the main thread
        for m in models_in_other_ctx:
            res.append(m.translate(self.ctx))
        # res = parallel_check_sat_multiprocessing(cnt_list, 4) # this one has bugs
        return res

    def parallel_check_process(self, cnt_list: List[z3.BoolRef]):
        raise NotImplementedError

    def build_mappings(self):
        """
        Build the mapping for replacement (not used for now)
        mappings = []
        for v in m:
            mappings.append((z3.BitVec(str(v)), v.size(), origin_ctx), z3.BitVecVal(m[v], v.size(), origin_ctx))
        """
        raise NotImplementedError


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
