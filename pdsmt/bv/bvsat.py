# coding: utf-8
"""
Flattening-based QF_BV solver
"""
import logging
import time

import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from z3.z3util import get_vars

from .mapped_blast import translate_smt2formula_to_cnf
from ..utils import SolverResult

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


class BVSolver:
    """
    Solving QF_BV formulas by combing Z3 and pySAT
      - Z3: Translate a QF_BV formula to a SAT formula
      - pySAT: solve the translated SAT formula
    """

    def __init__(self):
        self.fml = None
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat
        self.vars = []
        self.verbose = 0
        self.signed = False
        self.model = []

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula
        self.vars = get_vars(self.fml)

    def bit_blast(self):
        logger.debug("Start translating to CNF...")
        # NOTICE: can be slow
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.fml)
        self.bv2bool = bv2bool
        self.bool2id = id_table
        logger.debug("  from bv to bools: {}".format(self.bv2bool))
        logger.debug("  from bool to pysat id: {}".format(self.bool2id))

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        return clauses_numeric

    def check_sat(self):
        """Check satisfiability of a bit-vector formula"""
        clauses_numeric = self.bit_blast()
        # Main difficulty: how to infer signedness of each variable
        cnf = CNF(from_clauses=clauses_numeric)
        name = "minisat22"
        try:
            start_time = time.time()
            with Solver(name=name, bootstrap_with=cnf) as solver:
                if not solver.solve():
                    return SolverResult.UNSAT
                # TODO: figure out what is the order of the vars in the boolean model
                bool_model = solver.get_model()
                logger.debug("SAT solving time: {}".format(time.time() - start_time))
                self.model = bool_model
                return SolverResult.SAT
                """
                # The following code is for building the bit-vector model
                bv_model = {}
                if not self.signed: # unsigned
                    for bv_var in self.bv2bool:
                        bool_vars = self.bv2bool[bv_var]
                        start = self.bool2id[bool_vars[0]]  # start ID
                        bv_val = 0
                        for i in range(len(bool_vars)):
                            if bool_model[i + start - 1] > 0:
                                bv_val += 2 ** i
                        bv_model[bv_var] = bv_val
                else: # signed
                    # FIXME: the following seems to be wrong
                    for bv_var in self.bv2bool:
                        bool_vars = self.bv2bool[bv_var]
                        start = self.bool2id[bool_vars[0]]  # start ID
                        bv_val = 0
                        for i in range(len(bool_vars) - 1):
                            if bool_model[i + start - 1] > 0:
                                bv_val += 2 ** i
                        if bool_model[len(bool_vars) - 1 + start - 1] > 0:
                            bv_val = -bv_val
                        bv_model[bv_var] = bv_val
                # TODO: map back to bit-vector model
                self.model = bv_model
                print(bv_model)
                """
        except Exception as ex:
            print(ex)
