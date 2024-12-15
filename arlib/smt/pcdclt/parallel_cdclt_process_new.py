# coding: utf-8
"""Process-based Parallel CDCL(T)-style SMT Solving.

FIXME: generated by LLM. to check...
"""

import logging
from ctypes import c_char_p
from dataclasses import dataclass
from multiprocessing import Pool, Manager, cpu_count, Value
from typing import List, Optional

from arlib.bool import PySATSolver, simplify_numeric_clauses
from arlib.config import m_smt_solver_bin
from arlib.smt.pcdclt import SMTPreprocessor4Process, BooleanFormulaManager
from arlib.smt.pcdclt.exceptions import TheorySolverSuccess, PySMTSolverError
from arlib.smt.pcdclt.theory import SMTLibTheorySolver
from arlib.utils import SolverResult, parse_sexpr_string
from arlib.utils.exceptions import SMTLIBSolverError

logger = logging.getLogger(__name__)

# Configuration options
M_SIMPLIFY_BLOCKING_CLAUSES = True
DEFAULT_SAMPLE_SIZE = 10


@dataclass
class TheoryCheckResult:
    """Results from theory consistency checking."""
    is_consistent: bool
    unsat_core: Optional[str] = None
    error: Optional[Exception] = None


def check_theory_consistency(init_theory_fml: Value,
                             assumptions: List[str],
                             solver_bin: str) -> Optional[str]:
    """Check theory consistency of assumptions.

    Args:
        init_theory_fml: Initial theory formula (shared between processes)
        assumptions: List of Boolean variables to check
        solver_bin: Path to theory solver binary

    Returns:
        UNSAT core if theory-inconsistent, None if theory-consistent

    Raises:
        TheorySolverSuccess: If theory is consistent
        SMTLIBSolverError: If theory solver fails
    """
    logger.debug(f"Theory worker starting with solver: {solver_bin}")

    theory_solver = SMTLibTheorySolver(solver_bin)
    theory_solver.add(init_theory_fml.value)

    if theory_solver.check_sat_assuming(assumptions) == SolverResult.UNSAT:
        return theory_solver.get_unsat_core()

    raise TheorySolverSuccess()


def parse_raw_unsat_core(core: str, bool_manager: BooleanFormulaManager) -> List[int]:
    """
     Given an unsat core in string, build a blocking numerical clause from the core
    :param core: The unsat core built a theory solver
    :param bool_manager: The manger for tracking the information of Boolean abstraciton
         (e.g., the mapping between the name and the numerical ID)
    :return: The blocking clauses built from the unsat core
    """
    parsed_core = parse_sexpr_string(core)
    print(parsed_core)
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


def process_pysat_models(bool_models: List[List[int]],
                         bool_manager: BooleanFormulaManager) -> List[str]:
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


class ParallelTheorySolver:
    """Manages parallel theory solving using multiple processes."""

    def __init__(self, pool_size: Optional[int] = None):
        self.pool_size = pool_size or cpu_count()
        self.pool = Pool(processes=self.pool_size)

    def check_assumptions(self,
                          init_theory_fml: Value,
                          all_assumptions: List[List[str]]) -> List[str]:
        """Check multiple sets of assumptions in parallel.

        Args:
            init_theory_fml: Initial theory formula
            all_assumptions: List of assumption sets to check

        Returns:
            List of UNSAT cores or empty list if satisfiable
        """
        results = []
        for assumptions in all_assumptions:
            result = self.pool.apply_async(check_theory_consistency,
                                           (init_theory_fml, assumptions, m_smt_solver_bin))
            results.append(result)

        cores = []
        for result in results:
            try:
                core = result.get()
                if core:  # Non-empty core indicates UNSAT
                    cores.append(core)
            except TheorySolverSuccess:
                return []  # Found satisfiable assignment

        return cores

    def shutdown(self):
        """Clean up process pool resources."""
        self.pool.close()
        self.pool.join()


class CDCLTSolver:
    """Main CDCL(T) solving coordinator."""

    def __init__(self, sample_size: int = DEFAULT_SAMPLE_SIZE):
        self.sample_size = sample_size
        self.bool_solver = PySATSolver()
        self.theory_solver = ParallelTheorySolver()

    def _prepare_theory_formula(self, logic: str,
                                th_manager: BooleanFormulaManager) -> str:
        """Prepare theory formula with logic and signature."""
        return (f"(set-logic {logic}) "
                "(set-option :produce-unsat-cores true) "
                f"{' '.join(th_manager.smt2_signature)} "
                f"(assert {th_manager.smt2_init_cnt})")

    def _process_theory_cores(self,
                              raw_cores: List[str],
                              bool_manager: BooleanFormulaManager) -> List[List[int]]:
        """Process theory UNSAT cores into blocking clauses."""
        blocking_clauses = [parse_raw_unsat_core(core, bool_manager)
                            for core in raw_cores]

        if M_SIMPLIFY_BLOCKING_CLAUSES:
            blocking_clauses = simplify_numeric_clauses(blocking_clauses)
            logger.debug(f"Simplified blocking clauses: {blocking_clauses}")

        return blocking_clauses

    def solve(self, smt2string: str, logic: str) -> SolverResult:
        """Solve SMT formula using CDCL(T) algorithm.

        Args:
            smt2string: SMT2 format formula string
            logic: SMT logic name

        Returns:
            SolverResult indicating satisfiability
        """
        try:
            # Preprocessing
            preprocessor = SMTPreprocessor4Process()
            bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

            if preprocessor.status != SolverResult.UNKNOWN:
                return preprocessor.status

            # Initialize solvers
            self.bool_solver.add_clauses(bool_manager.numeric_clauses)
            theory_fml = self._prepare_theory_formula(logic, th_manager)

            with Manager() as manager:
                theory_fml_shared = manager.Value(c_char_p, theory_fml)

                while True:
                    if not self.bool_solver.check_sat():
                        return SolverResult.UNSAT

                    bool_models = self.bool_solver.sample_models(to_enum=self.sample_size)
                    assumptions = process_pysat_models(bool_models, bool_manager)

                    raw_cores = self.theory_solver.check_assumptions(
                        theory_fml_shared, assumptions)

                    if not raw_cores:  # Found satisfying assignment
                        return SolverResult.SAT

                    blocking_clauses = self._process_theory_cores(
                        raw_cores, bool_manager)
                    self.bool_solver.add_clauses(blocking_clauses)

        except (TheorySolverSuccess, SMTLIBSolverError, PySMTSolverError) as e:
            logger.error(f"Solver error: {str(e)}")
            return SolverResult.ERROR

        finally:
            self.theory_solver.shutdown()


def parallel_cdclt_process_new(smt2string: str,
                               logic: str,
                               num_samples_per_round: int = DEFAULT_SAMPLE_SIZE) -> SolverResult:
    """Main entry point for parallel CDCL(T) solving."""
    solver = CDCLTSolver(sample_size=num_samples_per_round)
    return solver.solve(smt2string, logic)
