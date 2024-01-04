# coding: utf-8
import logging
from typing import List

import z3

from arlib.optimization.qfbv_opt_blast import BitBlastOMTBVSolver

logger = logging.getLogger(__name__)


class BVOptimize:
    """
    NOTE: we Focus on boxed multi-objective OMT (lexixxorder and pareto not supported yet)
    """

    def __init__(self):
        self.fml = None
        self.verbose = 0

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula

    def maximize_with_maxsat(self, obj: z3.ExprRef, is_signed=False):
        """ TODO: add an option for selecting the engine of the MaxSAT
        """
        assert z3.is_bv(obj)
        sol = BitBlastOMTBVSolver()
        sol.from_smt_formula(self.fml)
        return sol.maximize_with_maxsat(obj, is_signed=is_signed)

    def maximize(self, obj: z3.ExprRef, is_signed=False):
        """TODO: integrate other engines and allow the users to choose
            - 1. Reduce to solving a quantified formula
            - 2. Solve using different engines of Z3
            - 3. Bit-vector level linear and binary search
            - 4. ....?"""
        return self.maximize_with_maxsat(obj, is_signed)

    def boxed_optimize(self, goals: List[z3.ExprRef], is_signed=False):
        """TODO: How to distinguish min goals and max goals in a list of ExperRef (the current API
            seems not to be a good idea)? A possible strategy is to convert each max goal
            to a min goal (or the reverse direction)"""
        results = []
        sol = BitBlastOMTBVSolver()
        sol.from_smt_formula(self.fml)
        for obj in goals:
            assert z3.is_bv(obj)
            # FIXME: currently, maximize_with_maxsat still calls bit-blasting many times.
            #  It is because that, we create a fresh variable for each  goao/objective,
            #   (e.g., we create an m to encode x - y), so that we can track the correlation of m and
            #  its corresponding Boolean variables.
            #   A simple "fixing strategy": create all the aux variables a the beginning
            results.append(sol.maximize_with_maxsat(obj, is_signed=is_signed))
        return results

    def lexicographic_optimize(self, goals):
        """TODO"""
        raise NotImplementedError

    def pareto_optimize(self, goals):
        """TODO"""
        raise NotImplementedError
