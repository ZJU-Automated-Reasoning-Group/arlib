"""
Forall Solver
"""
import logging
from enum import Enum
from typing import List

import concurrent.futures

import z3

from arlib.utils.exceptions import ForAllSolverSuccess
from arlib.efsmt.efbv.efbv_parallel_utils import parallel_check_sat_multiprocessing

logger = logging.getLogger(__name__)


def check_candidate(fml: List[z3.BoolRef], fml_ctx: z3.Context, origin_ctx: z3.Context):
    """
    :param fml: the formula to be checked
    :param fml_ctx: context of the fml
    :param origin_ctx: context of the main/origin thread
    :return A model in the origin_ctx
    """
    # print("Checking one ...", fml)
    solver = z3.SolverFor("QF_BV", ctx=fml_ctx)
    solver.add(fml)
    if solver.check() == z3.sat:
        m = solver.model()
        return m.translate(origin_ctx)  # to the original context?
    else:
        raise ForAllSolverSuccess()


def parallel_check_candidates(fmls: List[z3.BoolRef], num_workers: int):
    # Create new context for the computation
    # Note that we need to do this sequentially, as parallel access to the current context or its objects
    # will result in a segfault
    origin_ctx = fmls[0].ctx
    tasks = []
    for fml in fmls:
        # tasks.append((fml, main_ctx()))
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        tasks.append((i_fml, i_context))

    # TODO: try processes?
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #  with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_candidate, task[0], task[1], origin_ctx) for task in tasks]
        results = [f.result() for f in futures]
        return results


class FSolverMode(Enum):
    SEQUENTIAL = 0
    PARALLEL = 1  # parallel check
    EXTERNAL_PARALLEL = 2  # use external SMT solvers for parallel


class ForAllSolver(object):
    def __init__(self):
        self.solver_mode = FSolverMode.PARALLEL
        # self.solver_mode = FSolverMode.SEQUENTIAL
        self.forall_vars = []
        self.phi = None

    def push(self):
        return

    def pop(self):
        return

    def check(self, cnt_list: List[z3.BoolRef]):
        if self.solver_mode == FSolverMode.SEQUENTIAL:
            return self.sequential_check(cnt_list)
        elif self.solver_mode == FSolverMode.PARALLEL:
            return self.parallel_check(cnt_list)

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
                models.append(s.model())
            elif res == z3.unsat:
                return []  # at least one is UNSAT
        return models

    def parallel_check(self, cnt_list: List[z3.BoolRef]):
        """
        """
        res = parallel_check_candidates(cnt_list, 4)
        # res = parallel_check_sat_multiprocessing(cnt_list, 4) # this one has bugs
        return res


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
