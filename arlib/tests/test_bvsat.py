# coding: utf-8
"""
For testing the QF_BV solver
"""

from arlib.tests import TestCase, main
from arlib.bv.bv_sat import BVSolver


class TestBVSat(TestCase):
    """
    Test the bit-blasting based solver
    """

    def test_bvsat(self):
        import z3
        import logging
        logging.basicConfig(level=logging.DEBUG)
        x, y = z3.BitVecs("x y", 5)
        fml = z3.And(5 < x, x < y, y < 8)
        sol = BVSolver()
        sol.from_smt_formula(fml)
        print(sol.check_sat())


if __name__ == '__main__':
    main()
