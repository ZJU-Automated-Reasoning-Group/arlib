# coding: utf-8
import logging
import multiprocessing
from multiprocessing import cpu_count

from .bool import PySATSolver
from .exceptions import TheorySolverSuccess
from .preprocessing import SMTPreprocess
from .sexpr import parse_sexpr
from .theory import SMTLibTheorySolver
from .util import SolverResult

logger = logging.getLogger(__name__)


def check_theory_consistency(init_theory_fml, assumptions):
    """
    Check T-consistency
    :param init_theory_fml:
    :param assumptions: a list of Boolean variables
    :return: unsat core if T-inconsistency
    """
    theory_solver = SMTLibTheorySolver()
    theory_solver.add(init_theory_fml)
    if SolverResult.UNSAT == theory_solver.check_sat_assuming(assumptions):
        core = theory_solver.get_unsat_core()
        return core
    raise TheorySolverSuccess()
    # return ""  # empty core indicates SAT?


def theory_solve(init_theory_fml, all_assumptions, pool):
    """
    Call theory solvers
    """
    results = []
    for i in range(len(all_assumptions)):
        result = pool.apply_async(check_theory_consistency,
                                  (init_theory_fml, all_assumptions[i],))
        results.append(result)

    raw_unsat_cores = []
    for i in range(len(all_assumptions)):
        result = results[i].get()
        if result == "":  # empty core indicates SAT?
            return []
        raw_unsat_cores.append(result)

    return raw_unsat_cores


def parse_raw_unsat_core(core: str, bool_manager):
    """
    Given a unsat core in string,
    build a numerical clause
    """
    parsed_core = parse_sexpr(core)
    assert len(parsed_core) >= 1
    # Let the parsed_core be ['p@4', 'p@7', ['not', 'p@6']]
    blocking_clauses_core = []  # map the core to a numerical clause
    # the blocking clauses should be (not (and 4 7 -6)), i.e., [-4, -7, 6]
    for ele in parsed_core:
        if isinstance(ele, list):
            blocking_clauses_core.append(bool_manager.vars2num[ele[1]])
        else:
            blocking_clauses_core.append(-bool_manager.vars2num[ele])

    return blocking_clauses_core


def process_pysat_models(bool_models, bool_manager):
    all_assumptions = []
    for model in bool_models:
        assumptions = []
        for val in model:
            if val > 0:
                assumptions.append(bool_manager.num2vars[val])
            else:
                assumptions.append("(not {})".format(bool_manager.num2vars[abs(val)]))
        all_assumptions.append(assumptions)

    return all_assumptions


def parallel_cdclt(smt2string: str):
    """
    The entrance
    """
    preprocessor = SMTPreprocess()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    # term_signal = multiprocessing.Manager().Queue()  # flag for termination
    pool = multiprocessing.Pool(processes=cpu_count())  # process pool

    bool_solver = PySATSolver()
    bool_solver.add_clauses(bool_manager.numeric_clauses)

    init_theory_fml = " (set-logic ALL) " + " (set-option :produce-unsat-cores true) " \
                      + " ".join(th_manager.smt2_signature) + "(assert {})".format(th_manager.smt2_init_cnt)

    logger.debug("Finish initializing bool solvers")

    result = SolverResult.UNKNOWN
    sample_number = 5

    try:
        while True:
            is_sat = bool_solver.check_sat()
            if not is_sat:
                result = SolverResult.UNSAT
                break
            # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
            bool_models = bool_solver.sample_models(to_enum=sample_number)
            # print("bool models: ", bool_models)

            all_assumptions = process_pysat_models(bool_models, bool_manager)
            raw_unsat_cores = theory_solve(init_theory_fml, all_assumptions, pool)

            if len(raw_unsat_cores) == 0:
                result = SolverResult.SAT
                break

            logger.debug("raw unsat cores: {}".format(raw_unsat_cores))
            blocking_clauses = []
            for core in raw_unsat_cores:
                blocking_clauses.append(parse_raw_unsat_core(core, bool_manager))

            # print("blocking clauses: ", blocking_clauses)
            bool_solver.add_clauses(blocking_clauses)

    except TheorySolverSuccess:
        # print("Theory solver success!")
        result = SolverResult.SAT

    pool.close()
    pool.join()

    return result
