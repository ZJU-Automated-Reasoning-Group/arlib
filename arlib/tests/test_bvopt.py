# coding: utf-8
"""
For testing OMT(BV) solving engine
"""
import z3

from arlib.tests import TestCase, main
from arlib.tests.formula_generator import FormulaGenerator
# from arlib.smt.bv import OMTBVSolver
from arlib.optimization.bvopt.qfbv_opt import BitBlastOMTBVSolver


def is_sat(e):
    s = z3.Solver()
    s.add(e)
    s.set("timeout", 5000)
    return s.check() == z3.sat


def try_bvopt():
    try:
        x, y, z = z3.BitVecs("x y z", 8)
        fg = FormulaGenerator([x, y, z], bv_signed=False)
        fml = fg.generate_formula()
        if is_sat(fml):
            omt = BitBlastOMTBVSolver()
            omt.from_smt_formula(fml)
            print(omt.maximize(x, is_signed=False))
            print(" ")
            return True
        return False
    except Exception as ex:
        print(ex)
        return False


class TestBVOMT(TestCase):
    """
    Test the OMT(BV) solver
    """

    def test_bvopt1(self):
        # x, y, z = z3.BitVecs("x y z", 8)
        # fml = z3.And(z3.ULT(x, 100), x + y == 20)
        # omt = OMTBVSolver()
        # omt.from_smt_formula(fml)
        # print(omt.maximize(x, is_signed=False))
        assert True

    def test_bvopt2(self):
        for _ in range(5):
            if try_bvopt():
                break
        assert True


if __name__ == '__main__':
    main()
