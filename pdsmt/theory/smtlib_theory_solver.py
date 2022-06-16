# coding: utf-8
import logging
from typing import List
from ..config import m_smt_solver_bin
from ..smtlib_solver import SMTLIBSolver

logger = logging.getLogger(__name__)


class SMTLibTheorySolver(object):
    """
    Use smtlib_solver class to interact with a binary solver
    """

    def __init__(self):
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def __del__(self):
        self.bin_solver.stop()

    def add(self, smt2string: str):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        return self.bin_solver.check_sat()

    def check_sat_assuming(self, assumptions: List[str]):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        # cnts = "(assert ( and {}))\n".format(" ".join(assumptions))
        # self.add(cnts)
        # return self.check_sat()
        return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        return self.bin_solver.get_unsat_core()
