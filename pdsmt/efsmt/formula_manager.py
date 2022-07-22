# coding: utf-8
"""
For maintaining the correlations of bit-vec and Boolean world.
"""
import logging
from typing import List

import z3

from pdsmt.bv import translate_smt2formula_to_cnf

logger = logging.getLogger(__name__)


def is_inconsistent(fml_a, fml_b):
    s = z3.Solver()
    s.add(z3.And(fml_a, fml_b))
    return s.check() == z3.unsat


class BVFormulaManager:

    def __init__(self):
        self.universal_bools = []
        self.existential_bools = []
        self.bool_clauses = []

    def mapped_bit_blast(self, fml: z3.BoolRef, universal_vars: List[z3.ExprRef]):
        """" Translate a bit-vector formula to a Boolean formula
        """
        # TODO: should handle cases where fml is simplified to be true or false
        bv2bool, bool2id, header, clauses = translate_smt2formula_to_cnf(fml)
        logger.debug("  from bv to bools: {}".format(bv2bool))
        logger.debug("  from bool to sat id: {}".format(bool2id))

        for bv_var in universal_vars:
            # print(bv_var, ": corresponding bools")
            # print([id_table[bname] for bname in bv2bool[str(bv_var)]])
            for bool_var_name in bv2bool[str(bv_var)]:
                self.universal_bools.append(bool2id[bool_var_name])

        for cls in clauses:
            self.bool_clauses.append([int(lit) for lit in cls.split(" ")])
