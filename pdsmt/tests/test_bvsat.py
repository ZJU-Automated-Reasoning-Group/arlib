# coding: utf-8
from . import TestCase, main
from ..bv.bvsat import BVSolver


class TestBVSat(TestCase):

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
