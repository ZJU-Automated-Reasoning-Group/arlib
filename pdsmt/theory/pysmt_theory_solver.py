# coding: utf-8
import logging
from pysmt.logics import AUTO
# from pysmt.oracles import get_logic
# from pysmt.shortcuts import EqualsOrIff
from pysmt.shortcuts import Solver, Portfolio, get_model
from pysmt.shortcuts import Symbol, Bool, And, Not
from pysmt.typing import INT, REAL

"""
Wrappers for PySMT
"""

logger = logging.getLogger(__name__)


class PySMTTheorySolver(object):

    def __init__(self):
        self.solver = Solver()

    def add(self, smt2string):
        raise NotImplementedError

    def check_sat(self):
        raise NotImplementedError

    def check_sat_assuming(self, assumptions):
        raise NotImplementedError

    def get_unsat_core(self):
        raise NotImplementedError
