# coding: utf-8
"""
For testing the model counting engine
"""
import z3

from arlib.tests import TestCase, main
from arlib.counting.qfbv_counting import BVModelCounter


class TestBVCounter(TestCase):
    def test_bv_counger(self):
        mc = BVModelCounter()
        x = z3.BitVec("x", 4)
        y = z3.BitVec("y", 4)
        fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
        mc.init_from_fml(fml)
        # mc.count_model_by_bv_enumeration()
        mc.count_models_by_sharp_sat()


if __name__ == '__main__':
    main()

"""
def feat_test():
    mc = BVModelCounter()
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
    mc.init_from_fml(fml)
    # mc.init_from_file('../../benchmarks/t1.smt2')
    # mc.count_model_by_bv_enumeration()
    mc.count_models_by_sharp_sat()


if __name__ == '__main__':
    feat_test()
"""
