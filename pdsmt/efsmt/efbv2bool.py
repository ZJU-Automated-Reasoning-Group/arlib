# coding: utf-8
"""
Translate EFSMT(BV) to Boolean-level Problems
- QBF
- SAT
- BDD
- ?

"""
import logging

import z3
from ..bv.mapped_blast import translate_smt2formula_to_cnf


logger = logging.getLogger(__name__)


class EFSMT2Bool:

    def __init__(self, universal_vars: z3.ExprRef, fml: z3.BoolRef):
        self.bv_fml = fml  # a quantifier-free bit-vector formula
        self.universal_bv_vars = universal_vars  # the set of universal quantified variables

        self.cnf_numeric_clauses = []
        self.universal_bool_vars = []

        # self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        # self.bool2id = {}  # map a Boolean variable to its internal ID

    def flatten_to_bool(self):
        """
        Compute self.cnf_numeric_clauses and self.universal_bool_vars
          1. Bit-blast self.bv_fml to a CNF formula
          2. Track the Boolean variables corresponding to self.universal_bv_vars
        """
        logger.debug("Start translating to CNF...")
        # NOTICE: can be slow
        bv2bool, bool2id, header, clauses = translate_smt2formula_to_cnf(self.bv_fml)
        logger.debug("  from bv to bools: {}".format(bv2bool))
        logger.debug("  from bool to sat id: {}".format(bool2id))

        for bv_var in self.universal_bool_vars:
            for bool_var in bool2id[bv2bool[bv_var]]:
                self.universal_bool_vars.append(bool_var)

        for cls in clauses:
            self.cnf_numeric_clauses.append([int(lit) for lit in cls.split(" ")])

    def to_qbf(self):
        """
        EFSMT(BV) to QBF
         TODO: find some way to solve the QBF formula
          E.g., dump to a temporal QDIMACS file and call a bin solver?
        """
        self.flatten_to_bool()
        raise NotImplementedError

    def to_sat(self):
        """
        EFSMT(BV) to SAT
         TODO: find some way for QE (i.e., eliminate the universal quantified Boolean variables)
        """
        self. flatten_to_bool()
        raise NotImplementedError

    def to_bdd(self):
        """
        EFSMT(BV) to BDD
        :return:
        """
        raise NotImplementedError
