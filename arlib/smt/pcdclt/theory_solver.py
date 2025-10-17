"""Theory solver for CDCL(T) using SMT-LIB interface"""

import logging
import os
import time
from typing import List, Optional
from datetime import datetime

from arlib.utils.smtlib_solver import SMTLIBSolver
from arlib.utils import SolverResult
from arlib.smt.pcdclt.config import ENABLE_QUERY_LOGGING, QUERY_LOG_DIR

logger = logging.getLogger(__name__)


class TheorySolver:
    """Interface to external SMT solver for theory consistency checking

    Use as a context manager to ensure proper cleanup:
        with TheorySolver(solver_bin, worker_id) as solver:
            solver.add_formula(formula)
            result = solver.check_sat_assuming(assumptions)
    """

    def __init__(self, solver_bin: str, worker_id: Optional[int] = None):
        """
        Args:
            solver_bin: Path to SMT solver binary (e.g., "z3 -in")
            worker_id: ID for this worker (for logging)
        """
        self.solver = None
        self.worker_id = worker_id
        self.query_count = 0
        self._closed = False

        try:
            self.solver = SMTLIBSolver(solver_bin)
        except Exception as e:
            logger.error(f"Failed to initialize SMT solver: {e}")
            raise

        # Setup logging if enabled
        self.log_dir = None
        if ENABLE_QUERY_LOGGING and worker_id is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(QUERY_LOG_DIR, f"run_{timestamp}")
            os.makedirs(self.log_dir, exist_ok=True)
            self.start_time = time.time()
            logger.debug(f"Worker {worker_id} logging to {self.log_dir}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        self.close()
        return False

    def close(self):
        """Explicitly close and cleanup solver subprocess"""
        if self._closed:
            return

        self._closed = True
        if self.solver is not None:
            try:
                self.solver.stop()
                logger.debug(f"Theory solver {self.worker_id} subprocess cleaned up")
            except Exception as e:
                logger.warning(f"Error stopping solver subprocess: {e}")
            finally:
                self.solver = None

    def __del__(self):
        """Cleanup solver process (fallback, but context manager is preferred)"""
        if not self._closed:
            logger.warning(f"TheorySolver {self.worker_id} not properly closed, using __del__ cleanup")
            self.close()

    def add_formula(self, smt2_string: str):
        """Add formula constraints"""
        self.solver.assert_assertions(smt2_string)
        self._log_query("add", smt2_string)

    def check_sat_assuming(self, assumptions: List[str]) -> SolverResult:
        """
        Check satisfiability under assumptions

        Args:
            assumptions: List of literals (e.g., ['p@1', '(not p@2)'])

        Returns:
            SolverResult.SAT or SolverResult.UNSAT
        """
        logger.debug(f"Theory solver checking {len(assumptions)} assumptions")
        result = self.solver.check_sat_assuming(assumptions)
        self._log_query("check_sat_assuming", assumptions, result)
        return result

    def get_unsat_core(self) -> str:
        """Get unsat core from last UNSAT check (as s-expression string)"""
        core = self.solver.get_unsat_core()
        self._log_query("get_unsat_core", core)
        return core

    def _log_query(self, query_type: str, content, result=None):
        """Log query if logging is enabled"""
        if not ENABLE_QUERY_LOGGING or self.log_dir is None:
            return

        query_id = self.query_count
        self.query_count += 1

        # Write query to individual file
        filename = f"worker_{self.worker_id}_query_{query_id}.smt2"
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"; Worker: {self.worker_id}\n")
            f.write(f"; Query: {query_id}\n")
            f.write(f"; Type: {query_type}\n")
            f.write(f"; Time: {time.time() - self.start_time:.3f}s\n")

            if query_type == "check_sat_assuming":
                f.write(f"\n; Assumptions:\n")
                for assumption in content:
                    f.write(f"; {assumption}\n")
                f.write(f"\n(check-sat-assuming ({' '.join(content)}))\n")
            else:
                f.write(f"\n{content}\n")

            if result is not None:
                f.write(f"\n; Result: {result}\n")
