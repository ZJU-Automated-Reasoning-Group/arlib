# coding: utf-8
import logging
from pysmt.logics import AUTO
from pysmt.shortcuts import Solver, get_unsat_core
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import smtlibscript_from_formula
try:  # for Python2
    from cStringIO import StringIO
except ImportError:  # for Python3
    from io import StringIO


"""
Wrappers for PySMT
"""

logger = logging.getLogger(__name__)


class PySMTTheorySolver(object):

    def __init__(self):
        self.solver = Solver()

    def add(self, smt2string):
        parser = SmtLibParser()
        script = parser.get_script(StringIO(smt2string))
        fml = script.get_last_formula()
        self.solver.add_assertion(fml)

    def check_sat(self):
        return self.solver.solve()

    def check_sat_assuming(self, assumptions):
        raise NotImplementedError

    def get_unsat_core(self):
        return self.solver.get_unsat_core()