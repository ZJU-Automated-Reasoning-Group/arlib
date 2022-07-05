# coding: utf-8
"""
Use PySMT as the theory solver of the parallel CDCL(T) engine.
This will allow us to easily call the solvers supported by pySMT
Note that we only use it for dealing with a conjunction of formulas.
"""
import logging

from pysmt.shortcuts import Solver
from pysmt.smtlib.parser import SmtLibParser

try:  # for Python2
    from cStringIO import StringIO
except ImportError:  # for Python3
    from io import StringIO

logger = logging.getLogger(__name__)


class PySMTTheorySolver(object):

    def __init__(self):
        # TODO: do we need to explicitly manager the context of pySMT?
        self.solver = Solver()

    def add(self, smt2string: str):
        """Add an SMT-LIB2 string to self.solver"""
        parser = SmtLibParser()
        script = parser.get_script(StringIO(smt2string))
        fml = script.get_last_formula()
        self.solver.add_assertion(fml)

    def add_assertion_from_z3expr(self, expr):
        """In some cases, we may need this interface.
        However, since a z3 expr can be very expressive,
        we may need to check whether pysmt can handle it."""
        raise NotImplementedError

    def check_sat(self):
        """Check sat"""
        return self.solver.solve()

    def check_sat_assuming(self, assumptions):
        raise NotImplementedError

    def get_unsat_core(self):
        return self.solver.get_unsat_core()
