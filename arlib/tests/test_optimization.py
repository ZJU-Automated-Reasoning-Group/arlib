"""Test OMT Solvers"""

from pysmt.test import TestCase
from pysmt.test import main

from pysmt.shortcuts import GE, Int, Symbol, INT, LE, GT, REAL, Real
# from pysmt.shortcuts import BVType, BVUGE, BVSGE, BVULE, BVSLE, BVUGT, BVSGT, BVULT, BVSLT, BVZero, BVOne, BV
from pysmt.shortcuts import And, Plus, Minus, get_env
from pysmt.logics import QF_LIA, QF_LRA, QF_BV
from arlib.optimization.goal import MaximizationGoal, MinimizationGoal, \
    MinMaxGoal, MaxMinGoal, MaxSMTGoal

from arlib.optimization.optimizer import Optimizer
# from arlib.optimization.omt_exceptions import OMTUnboundedOptimizationError
from arlib.optimization.omt_factory import OMTFactory


class TestOptimization(TestCase):

    def test_minimization_basic(self):
        return True
        with Optimizer(environment=get_env(), logic=QF_LIA) as opt:
            x = Symbol("x", INT)
            min = MinimizationGoal(x)
            formula = GE(x, Int(5))
            opt.add_assertion(formula)
            model, cost = opt.optimize(min)


if __name__ == '__main__':
    main()
