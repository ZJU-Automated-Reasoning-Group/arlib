# coding: utf-8
"""
1. Translate a bit-vector optimization problem to a weighted MaxSAT problem,
2. Call a third-path MaxSAT solver

TODO:
- Need to track the relations between
  - bit-vector variable and boolean variables
  - boolean variables and the numbers in pysat CNF
"""
import logging
import random
import time
from typing import List

import z3
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from z3.z3util import get_vars

from .mapped_blast import translate_smt2formula_to_cnf
from ..bool import MaxSATSolver

logger = logging.getLogger(__name__)

sat_solvers_in_pysat = ['cadical',
                        'gluecard30',
                        'gluecard41',
                        'glucose30',
                        'glucose41',
                        'lingeling',
                        'maplechrono',
                        'maplecm',
                        'maplesat',
                        'minicard',
                        'mergesat3',
                        'minisat22',
                        'minisat-gh']


class OMTBVSolver:
    """
    NOTE: we Focus on boxed multi-objective OMT (lexixxorder and pareto not supported yet)
    """

    def __init__(self):
        self.fml = None
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat
        self.vars = []
        self.verbose = 0

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula
        self.vars = get_vars(self.fml)

    def bit_blast(self):
        logger.debug("Start translating to CNF...")
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.fml)
        self.bv2bool = bv2bool
        self.bool2id = id_table
        logger.debug("  from bv to bools: {}".format(self.bv2bool))
        logger.debug("  from bool to pysat id: {}".format(self.bool2id))

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        # print("  pysat clauses: ", clauses_numric)
        return clauses_numeric

    def check_sat(self):
        clauses_numeric = self.bit_blast()
        # TODO: map back to bit-vector model
        cnf = CNF(from_clauses=clauses_numeric)
        name = random.choice(sat_solvers_in_pysat)
        try:
            start = time.time()
            with Solver(name=name, bootstrap_with=cnf) as solver:
                res = solver.solve()
                logger.debug("outcome by {0}: {1}".format(name, res))
            logger.debug("SAT solving time: {}".format(time.time() - start))
        except Exception as ex:
            print(ex)

    def maximize_with_maxsat(self, obj: z3.ExprRef, is_signed=False):
        """
        Weighted-MaxSAT based OMT(BV)
        NOTE: some algorithms may use bit-level binary search, such as Nadel's algorithm
        """
        assert z3.is_bv(obj)
        objname = obj

        if obj not in self.vars:
            # print(obj, "is not a var in self.vars")
            objvars = get_vars(obj)
            for v in objvars:
                if v not in self.vars:
                    raise Exception(str(obj), "contains a var not in the hard formula")
                    # return
            # create a new variable to represent obj (a term, e.g., x + y)
            objname = z3.BitVec(str(obj), objvars[0].sort().size())
            self.fml = z3.And(self.fml, objname == obj)
            self.vars.append(objname)

        after_simp = z3.Tactic("simplify")(self.fml).as_expr()
        if z3.is_true(after_simp):
            logger.debug("the hard formula is a tautology (obj can be any value)")
            return
        elif z3.is_false(after_simp):
            logger.error("the hard formula is trivially unsat")
            return

        obj_str = str(objname)

        logger.debug("Start solving OMT(BV) by reducing to weighted Max-SAT...")
        clauses_numeric = self.bit_blast()
        wcnf = WCNF()
        wcnf.extend(clauses_numeric)
        total_score = 0
        boovars = self.bv2bool[obj_str]
        if is_signed:
            for i in range(len(boovars) - 1):
                wcnf.append([self.bool2id[boovars[i]]], weight=2 ** i)
                total_score += 2 ** i
            wcnf.append([self.bool2id[boovars[len(boovars) - 1]]], weight=-(2 ** (len(boovars) - 1)))
            total_score -= 2 ** (len(boovars) - 1)
        else:
            for i in range(len(boovars)):
                wcnf.append([self.bool2id[boovars[i]]], weight=2 ** i)
                total_score += 2 ** i
        # print("hard: {}\nsoft: {}\nweight: {}\n".format(wcnf.hard, wcnf.soft, wcnf.wght))

        logger.debug("Start solving weighted Max-SAT via pySAT...")
        maxsat_sol = MaxSATSolver(wcnf)

        # 1. Use an existing weighted MaxSAT solving algorithm
        start = time.time()
        maxsat_sol.set_maxsat_engine("FM")
        cost = maxsat_sol.solve_wcnf()
        logger.debug("maximum of {0}: {1} ".format(obj_str, total_score - cost))
        logger.debug("FM MaxSAT time: {}".format(time.time() - start))

        # 2. Use binary-search-based MaxSAT solving (specialized for OMT(BV))
        start = time.time()
        assumption_lits = maxsat_sol.tacas16_binary_search()
        assumption_lits.reverse()
        sum_score = 0
        if is_signed:
            for i in range(len(assumption_lits) - 1):
                if assumption_lits[i] > 0:
                    sum_score += 2 ** i
            # 符号位是1表示负数,是0表示正数?
            if assumption_lits[-1] > 0:
                sum_score = -sum_score
        else:
            for i in range(len(assumption_lits)):
                if assumption_lits[i] > 0:
                    sum_score += 2 ** i
        logger.debug("\nmaximum of {0}: {1}".format(obj_str, sum_score))
        logger.debug("TACAS16 MaxSAT time: {}".format(time.time() - start))

        return sum_score

    """
    NOTE: the following ones are the external interface
    """

    def maximize(self, obj: z3.ExprRef, is_signed=False):
        return self.maximize_with_maxsat(obj, is_signed)

    def maximize_boxed(self, objs: List[z3.ExprRef], is_signed=False):
        res = []
        for obj in objs:
            res.append(self.maximize(obj, is_signed))
