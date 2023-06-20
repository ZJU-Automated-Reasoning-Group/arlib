# coding: utf-8
"""
Use Z3 as the theory solver of of the parallel CDCL(T) engine
Note: this is only used for multi-process communicating (where the formulas are passed as strings)
"""
import logging
from typing import List

import z3

logger = logging.getLogger(__name__)


class Z3TheorySolver(object):

    def __init__(self, logic: str = None):
        if logic:
            self.z3_solver = z3.SolverFor(logic)
        else:
            self.z3_solver = z3.Solver()

    def add(self, smt2string: str):
        self.z3_solver.add(z3.And(z3.parse_smt2_string(smt2string)))

    def check_sat(self):
        """TODO: make the return type of this function consistent"""
        logger.debug("Theory solver working...")
        return self.z3_solver.check()

    def check_sat_assuming(self, assumptions: List[str]):
        """
        For checking under assumptions
        TODO: if we parse the smt2string formulas at self.add,
          the self.z3_solver may not be able to understand the meanings of assumptions
        """
        logger.debug("Theory solver working...")
        # cnts = "(assert ( and {}))\n".format(" ".join(assumptions))
        # self.add(cnts)
        # return self.check_sat()
        raise NotImplementedError

    def get_unsat_core(self):
        """TODO: make the return type of this function consistent"""
        return self.z3_solver.unsat_core()
