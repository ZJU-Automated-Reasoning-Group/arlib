# coding: utf-8
"""
For testing OMT(BV) solving engine
"""
import z3

from . import TestCase, main
from .formula_generator import FormulaGenerator
from ..bv.bvopt import OMTBVSolver


def is_sat(e):
    s = z3.Solver()
    s.add(e)
    s.set("timeout", 5000)
    return s.check() == z3.sat


def try_bvopt():
    try:
        w, x, y, z = z3.BitVecs("w x y z", 8)
        fg = FormulaGenerator([x, y, z])
        fml = fg.generate_formula()
        if is_sat(fml):
            omt = OMTBVSolver()
            omt.from_smt_formula(fml)
            print(omt.maximize(x, is_signed=True))
            return True
        return False
    except Exception as ex:
        print(ex)
        return False


class TestBVOMT(TestCase):
    """
    Test the OMT(BV) solver
    """

    def test_bvopt(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)
        for _ in range(5):
            if try_bvopt():
                break
        assert True


if __name__ == '__main__':
    main()
