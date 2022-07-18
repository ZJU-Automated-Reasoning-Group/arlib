# coding: utf-8
"""
For testing model counting of bit-vector formulas
"""
import os

import z3

from pdsmt.tests import TestCase, main
from pdsmt.bv.bvcounting import BVModelCounter


def clear_tmp_cnf_files():
    if os.path.isfile('/tmp/out.cnf'):
        os.remove('/tmp/out.cnf')


class TestBVModelCounter(TestCase):
    """
    Test the model counter for QF_BV formulas
    """

    def test_model_counter(self):
        for _ in range(1):
            x = z3.BitVec("x", 4)
            y = z3.BitVec("y", 4)
            fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
            counter = BVModelCounter()
            print(counter.count_models(fml))
            clear_tmp_cnf_files()


if __name__ == '__main__':
    main()
