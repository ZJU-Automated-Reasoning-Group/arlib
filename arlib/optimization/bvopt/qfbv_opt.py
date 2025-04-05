# coding: utf-8
import logging
from typing import List

import z3

from arlib.optimization.bvopt.qfbv_opt_blast import BitBlastOMTBVSolver

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

    def check_satisfiability(self):
        """Check if the current formula is satisfiable"""
        clauses_numeric = self.bit_blast()
        cnf = CNF(from_clauses=clauses_numeric)
        name = random.choice(sat_solvers_in_pysat)

        try:
            with Solver(name=name, bootstrap_with=cnf) as solver:
                if solver.solve():
                    return True, solver.get_model()
                return False, None
        except Exception as ex:
            logger.error(f"SAT solving failed: {ex}")
            return False, None

    def maximize_with_maxsat(self, obj: z3.ExprRef, is_signed=False):
        """Maximize a bit-vector objective using MaxSAT-based optimization"""
        # First check if formula is satisfiable
        is_sat, _ = self.check_satisfiability()
        if not is_sat:
            logger.debug("Formula is unsatisfiable")
            return None

        try:
            # Original optimization logic
            clauses_numeric = self.bit_blast()
            # ... rest of the existing optimization code ...

        except Exception as ex:
            logger.error(f"Optimization failed: {ex}")
            return None

    def maximize(self, obj: z3.ExprRef, is_signed=False):
        """Wrapper for maximize_with_maxsat with additional error handling"""
        if self.fml is None:
            logger.error("No formula provided")
            return None

        try:
            return self.maximize_with_maxsat(obj, is_signed)
        except Exception as ex:
            logger.error(f"Maximization failed: {ex}")
            return None

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
