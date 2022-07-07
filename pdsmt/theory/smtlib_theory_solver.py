# coding: utf-8
"""
Use one or more binary solvers as the theory solver of the parallel CDCL(T) engine.
Note that we only use it for dealing with a conjunction of formulas.
"""
import logging
from typing import List

from ..utils.smtlib_solver import SMTLIBSolver, SMTLIBPortfolioSolver

logger = logging.getLogger(__name__)


class SMTLibTheorySolver(object):
    """
    Use smtlib_solver class to interact with a binary solver
    """

    def __init__(self, solver_bin):
        self.bin_solver = SMTLIBSolver(solver_bin)

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


class SMTLibPortfolioTheorySolver(object):
    """
    Use smtlib_solver class to interact with a binary solver
    TODO: test this (as we have not tried SMTLIBPortfolioSolver before)
    """

    def __init__(self, solvers_list: List[str]):
        solvers = solvers_list
        self.bin_solvers = SMTLIBPortfolioSolver(solvers)

    def __del__(self):
        self.bin_solvers.stop()

    def add(self, smt2string: str):
        self.bin_solvers.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        return self.bin_solvers.check_sat()

    def check_sat_assuming(self, assumptions: List[str]):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        cnts = "(assert ( and {}))\n".format(" ".join(assumptions))
        self.add(cnts)
        return self.check_sat()
        # return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        return self.bin_solvers.get_unsat_core()
