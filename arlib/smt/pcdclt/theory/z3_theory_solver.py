"""
Use Z3 as the theory solver for the parallel CDCL(T) engine (string-based IPC).
"""
import logging
from typing import List

import z3

logger = logging.getLogger(__name__)


class Z3TheorySolver(object):
    """Z3-based theory solver used in multi-process mode with string I/O."""

    def __init__(self, logic: str = None):
        self.z3_solver = z3.SolverFor(logic) if logic else z3.Solver()

    def add(self, smt2string: str):
        self.z3_solver.add(z3.And(z3.parse_smt2_string(smt2string)))

    def check_sat(self):
        logger.debug("Theory solver working...")
        return self.z3_solver.check()

    def check_sat_assuming(self, assumptions: List[str]):
        """Check satisfiability under assumptions."""
        logger.debug("Theory solver working...")
        z3_assumptions = [z3.parse_smt2_string(assumption)[0] for assumption in assumptions]
        return self.z3_solver.check(z3_assumptions)

    def get_unsat_core(self):
        return self.z3_solver.unsat_core()
