# coding: utf-8
"""
Process-based Parallel CDCL(T)-style SMT Solving
"""

import logging
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Manager
from ctypes import c_char_p
from typing import List

from arlib.cdclt import SMTPreprocessor4Process, BooleanFormulaManager
from arlib.bool import PySATSolver, simplify_numeric_clauses
from arlib.theory import SMTLibTheorySolver
from arlib.utils import SolverResult, parse_sexpr_string
from arlib.cdclt.exceptions import TheorySolverSuccess, PySMTSolverError
from arlib.utils.exceptions import SMTLIBSolverError
from arlib.config import m_smt_solver_bin

logger = logging.getLogger(__name__)

"""

# Some options to be configured (assuming use Z3 for now
"""
m_simplify_blocking_clauses = True


# End of options


def check_theory_consistency(init_theory_fml,
                             assumptions: List[str], bin_solver):
    """
    TODO: this function should be able to take a set of assumptions?
    Check T-consistency
    :param init_theory_fml: the initial theory formula (it is not a good idea
         to pass it everytime when we call this function, because init_theory_fml is const
    :param assumptions: a list of Boolean variables
    :param bin_solver: the binary solver
    :return: unsat core if T-inconsistent; raise TheorySolverSuccess if T-consistent
    """
    logger.debug("One theory worker starts: {}".format(bin_solver))
    theory_solver = SMTLibTheorySolver(bin_solver)
    theory_solver.add(init_theory_fml.value)
    if SolverResult.UNSAT == theory_solver.check_sat_assuming(assumptions):
        core = theory_solver.get_unsat_core()
        return core
    raise TheorySolverSuccess()
    # return ""  # empty core indicates SAT?


def theory_solve(init_theory_fml,
                 all_assumptions: List[str], pool: multiprocessing.Pool):
    """
    Call theory solvers to solve a set of assumptions.
    :param init_theory_fml: The theory formula encoding the mapping between
            Boolean variables and the theory atoms they encode
            (e.g., b1 = x >= 3, b2 = y <= 5, where b1 and b2 are Boolean variables)
    :param all_assumptions: The set of assumptions to be checked,
            (e.g., [[b1, b2], [b1], [not b1, b2]]
    :param pool: The process pool for parallel solving
    :return: The set of unsat cores (given by the theory solvers)
            Note that the theory solvers may throw an exception TheorySolverSuccess,
    """
    results = []
    # TODO: If len(all_assumptions) == 1, then there is only one model to check.
    #   For such cases, we may use a portfolio mode (TBD)
    """
    if len(all_assumptions) == 1:
        logger.debug("Only one Boolean model to decide: try portfolio")
        for i in range(4):
            print(m_portfolio_solvers[i])
            result = pool.apply_async(check_theory_consistency,
                                      (init_theory_fml, all_assumptions[0], m_portfolio_solvers[i]))
            results.append(result)
    else:
        for i in range(len(all_assumptions)):
            result = pool.apply_async(check_theory_consistency,
                                      (init_theory_fml, all_assumptions[i], m_smt_solver_bin + " -in "))
            results.append(result)
    """
    for i in range(len(all_assumptions)):
        result = pool.apply_async(check_theory_consistency,
                                  (init_theory_fml, all_assumptions[i], m_smt_solver_bin))
        results.append(result)

    raw_unsat_cores = []
    for i in range(len(all_assumptions)):
        result = results[i].get()
        if result == "":  # empty core indicates SAT?
            return []  # if it is true, we may need to raise TheorySolverSuccess
        raw_unsat_cores.append(result)

    return raw_unsat_cores


def parse_raw_unsat_core(core: str, bool_manager: BooleanFormulaManager) -> List[int]:
    """
     Given an unsat core in string, build a blocking numerical clause from the core
    :param core: The unsat core built a theory solver
    :param bool_manager: The manger for tracking the information of Boolean abstraciton
         (e.g., the mapping between the name and the numerical ID)
    :return: The blocking clauses built from the unsat core
    """
    parsed_core = parse_sexpr_string(core)
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


def process_pysat_models(bool_models: List[List[int]], bool_manager: BooleanFormulaManager) -> List[str]:
    """
    Given a set of Boolean models, built the assumptions (to be checked by the theory solvers)
    :param bool_models: The set of Boolean models
    :param bool_manager: The manger for tracking the information of Boolean abstraciton
         (e.g., the mapping between the name and the numerical ID)
    :return: The set of assumptions
    """
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


def parallel_cdclt_process(smt2string: str, logic: str) -> SolverResult:
    """
    The main function of the parallel CDCL(T) SMT solving enigne
    :param smt2string: The formula to be solved
    :param logic: The logic type
    :return: The satisfiability result
    """
    preprocessor = SMTPreprocessor4Process()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    pool = multiprocessing.Pool(processes=cpu_count())  # process pool

    bool_solver = PySATSolver()
    bool_solver.add_clauses(bool_manager.numeric_clauses)

    init_theory_fml_str = " (set-logic {}) ".format(logic) + " (set-option :produce-unsat-cores true) " \
                          + " ".join(th_manager.smt2_signature) + "(assert {})".format(th_manager.smt2_init_cnt)

    # print(init_theory_fml_str)

    """
    According to https://stackoverflow.com/questions/17377426/shared-variable-in-pythons-multiprocessing
    It seems Manger is slower than Value, but we cannot directly use Value for string? 
    """
    manager = Manager()
    init_theory_fml_str_shared = manager.Value(c_char_p, init_theory_fml_str)

    logger.debug("Finish initializing Bool solvers")

    sample_number = 10
    try:
        while True:
            is_sat = bool_solver.check_sat()
            if not is_sat:
                result = SolverResult.UNSAT
                break
            # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
            logger.debug("Boolean abstraction is satisfiable")
            bool_models = bool_solver.sample_models(to_enum=sample_number)
            logger.debug("Finish sampling Boolean models; Start checking T-consistency!")

            all_assumptions = process_pysat_models(bool_models, bool_manager)
            raw_unsat_cores = theory_solve(init_theory_fml_str_shared, all_assumptions, pool)

            if len(raw_unsat_cores) == 0:
                result = SolverResult.SAT
                break

            logger.debug("Theory solvers finished; Raw unsat cores: {}".format(raw_unsat_cores))
            blocking_clauses = []
            for core in raw_unsat_cores:
                blocking_clauses.append(parse_raw_unsat_core(core, bool_manager))

            # TODO: simplify the blocking clauses
            logger.debug("Blocking clauses from cores: {}".format(blocking_clauses))
            if m_simplify_blocking_clauses:
                blocking_clauses = simplify_numeric_clauses(blocking_clauses)
                logger.debug("Simplified blocking clauses: {}".format(blocking_clauses))

            bool_solver.add_clauses(blocking_clauses)

    except TheorySolverSuccess:
        # print("One theory solver success!!")
        result = SolverResult.SAT
    except SMTLIBSolverError as ex:
        print(ex)
        result = SolverResult.ERROR
    except PySMTSolverError as ex:
        print(ex)
        result = SolverResult.ERROR
    # except Exception as ex:
    #    result = SolverResult.ERROR

    pool.close()
    pool.join()

    return result
