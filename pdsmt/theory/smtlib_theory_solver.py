# coding: utf-8
import logging
from ..smtlib_solver import SMTLIBSolver
from ..config import m_smt_solver_bin

logger = logging.getLogger(__name__)


class SMTLibTheorySolver(object):

    def __init__(self):
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def __del__(self):
        self.bin_solver.stop()

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        return self.bin_solver.check_sat()

    def check_sat_assuming(self, assumptions):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        return self.bin_solver.get_unsat_core()
