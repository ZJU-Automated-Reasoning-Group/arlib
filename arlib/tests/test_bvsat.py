# coding: utf-8
"""
For testing the QF_BV solver
"""

from arlib.tests import TestCase, main
from arlib.smt.bv import QFBVSolver


class TestBVSat(TestCase):
    """
    Test the bit-blasting based solver
    """

    def test_bvsat(self):
        import z3
        x, y = z3.BitVecs("x y", 5)
        fml = z3.And(5 < x, x < y, y < 8)
        sol = QFBVSolver()
        print(sol.solve_smt_formula(fml))
        # s = z3.Solver()
        # s.add(fml)
        # print(s.to_smt2())


if __name__ == '__main__':
    main()
