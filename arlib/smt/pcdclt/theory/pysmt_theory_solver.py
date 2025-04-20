# coding: utf-8
"""
Use PySMT as the theory solver of the parallel CDCL(T) engine.

This will allow us to easily call the solvers supported by pySMT.
Note that we only use it for dealing with a conjunction of formulas.
"""

import logging
from typing import List

from pysmt.shortcuts import Solver, And, TRUE, FALSE
from pysmt.smtlib.parser import SmtLibParser
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.fnode import FNode

try:  # for Python2
    from cStringIO import StringIO
except ImportError:  # for Python3
    from io import StringIO

logger = logging.getLogger(__name__)


class PySMTTheorySolver:
    def __init__(self):
        self.solver = Solver()

    def add(self, smt2string: str):
        """Add an SMT-LIB2 string to self.solver"""
        parser = SmtLibParser()
        script = parser.get_script(StringIO(smt2string))
        fml = script.get_last_formula()
        self.solver.add_assertion(fml)

    def add_assertion(self, assertion: FNode):
        """Add a PySMT formula assertion to the solver."""
        self.solver.add_assertion(assertion)

    def check_sat(self) -> bool:
        """Check satisfiability of the assertions in the solver."""
        try:
            return self.solver.solve()
        except SolverReturnedUnknownResultError:
            logger.warning("Solver returned unknown result. Assuming UNSAT.")
            return False

    def check_sat_assuming(self, assumptions: List[FNode]) -> bool:
        """Check satisfiability of the assertions assuming the given assumptions."""
        try:
            return self.solver.solve(assumptions)
        except SolverReturnedUnknownResultError:
            logger.warning("Solver returned unknown result. Assuming UNSAT.")
            return False

    def get_unsat_core(self) -> List[FNode]:
        """Get the unsat core after an UNSAT check_sat call."""
        return self.solver.get_unsat_core()

    def reset(self):
        """Reset the solver, removing all assertions."""
        self.solver.reset_assertions()

    def push(self):
        """Push the current context of the solver."""
        self.solver.push()

    def pop(self):
        """Pop the current context of the solver."""
        self.solver.pop()
