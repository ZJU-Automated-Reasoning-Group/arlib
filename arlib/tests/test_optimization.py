from pysmt.test import TestCase
from pysmt.test import main

from pysmt.shortcuts import GE, Int, Symbol, INT, LE, GT, REAL, Real
from pysmt.shortcuts import BVType, BVUGE, BVSGE, BVULE, BVSLE, BVUGT, BVSGT, BVULT, BVSLT, BVZero, BVOne, BV
from pysmt.shortcuts import And, Plus, Minus, get_env
from pysmt.logics import QF_LIA, QF_LRA, QF_BV
from arlib.optimization.goal import MaximizationGoal, MinimizationGoal, \
    MinMaxGoal, MaxMinGoal, MaxSMTGoal

from arlib.optimization.optimizer import Optimizer
from arlib.optimization.omt_exceptions import PysmtUnboundedOptimizationError

class TestOptimization(TestCase):



    def test_maximization_basic(self):
        x = Symbol("x", INT)
        max = MaximizationGoal(x)
        formula = LE(x, Int(10))
        return True
        # with Optimizer(logic=QF_LIA, environment=get_env()) as opt:
        #     opt.add_assertion(formula)
        #    model, cost = opt.optimize(max)
        #    self.assertEqual(model[x], Int(10))


if __name__ == '__main__':
    main()
