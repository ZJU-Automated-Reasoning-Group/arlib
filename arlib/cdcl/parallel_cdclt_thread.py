# coding: utf-8
"""
Process-based Parallel CDCL(T)-style SMT Solving
"""

import itertools
import logging
import multiprocessing
import concurrent.futures
from typing import List

import z3

from arlib.utils import SolverResult
from arlib.cdcl import SMTPreprocessor4Thread
from arlib.cdcl.exceptions import TheorySolverSuccess, TheorySolverError

logger = logging.getLogger(__name__)

"""

# Some options to be configured (assuming use Z3 for now
"""
m_simplify_blocking_clauses = True
m_logic = "ALL"


# End of options


def check_theory_consistency(init_theory_fml: z3.BoolRef, assumptions: List[z3.ExprRef]):
    """
    """
    logger.debug("One theory worker starts")
    theory_solver = z3.Solver(ctx=init_theory_fml.ctx)
    theory_solver.set(unsat_core=True)
    theory_solver.add(init_theory_fml)
    res = theory_solver.check(assumptions)
    if res == z3.unsat:
        core = theory_solver.unsat_core()
        return core
    elif res == z3.sat:
        raise TheorySolverSuccess()
    else:
        raise TheorySolverError()
    # return ""  # empty core indicates SAT?


def theory_solve(init_theory_fml: z3.ExprRef, all_assumptions: List[List[z3.BoolRef]], num_workers: int):
    """
    Call theory solvers to solve a set of assumptions.
    :param init_theory_fml: The theory formula encoding the mapping between
            Boolean variables and the theory atoms they encode
            (e.g., b1 = x >= 3, b2 = y <= 5, where b1 and b2 are Boolean variables)
    :param all_assumptions: The set of assumptions to be checked,
            (e.g., [[b1, b2], [b1], [not b1, b2]]
    :return: The set of unsat cores (given by the theory solvers)
            Note that the theory solvers may throw an exception TheorySolverSuccess,
    """
    tasks = []
    # TODO: Creating new contexts and translating init_theory_fml to those contexts repeatedly
    #  can be time-consuming, as the translation needs to recursively explore the AST.
    #  We may create a pool of contexts in the main function,
    #  and reuse them everytime we need to perform the theory solving.
    #  Particularly, in our case, the init_theory_fml is always the same for all queries
    for assumptions in all_assumptions:
        i_context = z3.Context()
        i_fml = init_theory_fml.translate(i_context)
        i_assumptions = [lit.translate(i_context) for lit in assumptions]
        tasks.append((i_fml, i_assumptions))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #  with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_theory_consistency, task[0], task[1]) for task in tasks]
        unsat_cores_from_other_contexts = [f.result() for f in futures]
        unsat_cores = []  # translate the model to the main thread
        for core in unsat_cores_from_other_contexts:
            unsat_cores.append(core.translate(init_theory_fml.ctx))
        return unsat_cores


class BooleanSolver:
    """
    TODO: replace this class with a parallel uniform sampler
    """

    def __init__(self, bool_fml: z3.ExprRef, bool_vars: List[z3.ExprRef]):
        self.bool_abstraction = [bool_fml]
        self.variables = bool_vars

    def check_sat(self):
        solver = z3.Solver()
        solver.add(z3.And(self.bool_abstraction))
        return solver.check()

    def sample_models(self, to_sample: int):
        solver = z3.Solver()
        solver.add(z3.And(self.bool_abstraction))
        bool_models = []
        solutions = 0
        for assignment in itertools.product(*[(x, z3.Not(x)) for x in self.variables]):  # all combinations
            if solver.check(assignment) == z3.sat:  # conditional check (does not add assignment permanently)
                bool_models.append(solver.model())
                solutions = solutions + 1
                if solutions == to_sample:
                    break
        return bool_models

    def add_clauses(self, clauses):
        self.bool_abstraction.append(z3.And(clauses))


def parallel_cdclt_thread(smt2string: str, logic: str) -> SolverResult:
    """
    The main function of the parallel CDCL(T) SMT solving enigne
    :param smt2string: The formula to be solved
    :param logic: The logic type
    :return: The satisfiability result
    """
    preprocessor = SMTPreprocessor4Thread()
    preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    global m_logic
    m_logic = logic

    bool_solver = BooleanSolver(preprocessor.bool_abstraction, preprocessor.bool_variables)
    init_theory_fml = preprocessor.init_theory_fml

    logger.debug("Finish initializing Bool solvers")

    num_workers = multiprocessing.cpu_count()
    sample_number = 5
    try:
        round = 1
        while True:
            logger.debug("Round: {}".format(round))
            if bool_solver.check_sat() == z3.unsat:
                result = SolverResult.UNSAT
                break
            # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
            #  Perhaps we should...
            logger.debug("Boolean abstraction is satisfiable")
            bool_models = bool_solver.sample_models(to_sample=sample_number)
            logger.debug("Finish sampling Boolean models; Start checking T-consistency!")

            all_assumptions = []
            # logger.debug("Sampled Boolean models: {}".format(bool_models))
            for model in bool_models:
                assumptions = [b if z3.is_true(model.eval(b)) else z3.Not(b) for b in preprocessor.bool_variables]
                all_assumptions.append(assumptions)

            # logger.debug("Assumptions from Boolean models: {}".format(all_assumptions))
            unsat_cores = theory_solve(init_theory_fml, all_assumptions, num_workers=num_workers)
            if len(unsat_cores) == 0:
                result = SolverResult.SAT
                break

            logger.debug("Theory solvers finished; Unsat cores: {}".format(unsat_cores))
            blocking_clauses = []
            for core in unsat_cores:
                blocking_clause = z3.Or([z3.Not(b) for b in core])
                blocking_clauses.append(blocking_clause)

            # TODO: simplify the blocking clauses
            blocking = z3.simplify(z3.And(blocking_clauses))
            logger.debug("Blocking clauses from cores: {}".format(blocking))
            bool_solver.add_clauses(blocking)
            round += 1

    except TheorySolverSuccess:
        # print("One theory solver success!!")
        result = SolverResult.SAT
    # except Exception as ex:
    #    result = SolverResult.ERROR

    return result
