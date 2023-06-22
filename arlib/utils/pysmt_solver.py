# coding: utf-8
"""
Augmenting Z3 using PySMT
"""

import logging
import z3
from pysmt.logics import QF_BV, QF_UFBV, QF_ABV, QF_AUFBV, \
    QF_LIA, QF_LRA, QF_UFLIA, QF_UFLRA
# from pysmt.oracles import get_logic
from pysmt.shortcuts import Solver

logger = logging.getLogger(__name__)

name2logic = {'QF_BV': QF_BV,
              'QF_UFBV': QF_UFBV,
              'QF_AUFBV': QF_AUFBV,
              'QF_ABV': QF_ABV,
              'QF_LIA': QF_LIA,
              "QF_LRA": QF_LRA,
              "QF_UFLIA": QF_UFLIA,
              "QF_UFLRA": QF_UFLRA
              }


# NOTE: both pysmt and z3 have a class "Solver"

class PySMTSolver(z3.Solver):

    def __init__(self, debug=False):
        super(PySMTSolver, self).__init__()

    @staticmethod
    def convert(zf: z3.ExprRef):
        """
        FIXME: if we do not call "pysmt_vars = ...", z3 will report naming warning..
        """
        z3s = Solver(name='z3')
        pysmt_fml = z3s.converter.back(zf)
        return pysmt_fml

    def check_with_pysmt(self, logic: str):
        """TODO: build a Z3 model?"""
        z3fml = z3.And(self.assertions())
        pysmt_fml = PySMTSolver.convert(z3fml)
        # print(pysmt_vars)
        try:
            f_logic = name2logic[logic]
            with Solver(logic=f_logic) as solver:
                solver.add_assertion(pysmt_fml)
                res = solver.solve()
                if res:
                    return z3.sat
                return z3.unsat
        except Exception:
            return z3.unknown

def test():
    x, y, z = z3.Ints("x y z")
    fml = z3.And(x > 10, y < 19, z == 3.0)
    sol = PySMTSolver()
    sol.add(fml)
    print(sol.check())
    # sol.all_smt([x, y])

# test()
