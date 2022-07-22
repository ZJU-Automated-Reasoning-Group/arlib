# coding: utf-8
"""
For maintaining the correlations of bit-vec and Boolean world.
"""
import logging
from typing import List

import z3

from pdsmt.bv import translate_smt2formula_to_cnf

logger = logging.getLogger(__name__)


class EFBVFormulaManager:

    def __init__(self):
        self.universal_bools = []
        self.existential_bools = []
        self.bool_clauses = []

    def initialize(self, fml: z3.BoolRef, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef]):
        """" Translate a bit-vector formula to a Boolean formula and initialize some self.fields
        """
        # TODO: should handle cases where fml is simplified to be true or false
        bv2bool, bool2id, header, clauses = translate_smt2formula_to_cnf(fml)
        logger.debug("  from bv to bools: {}".format(bv2bool))
        logger.debug("  from bool to sat id: {}".format(bool2id))

        for bv_var in existential_vars:
            for bool_var_name in bv2bool[str(bv_var)]:
                self.existential_bools.append(bool2id[bool_var_name])

        for bv_var in universal_vars:
            # print(bv_var, ": corresponding bools")
            # print([id_table[bname] for bname in bv2bool[str(bv_var)]])
            for bool_var_name in bv2bool[str(bv_var)]:
                self.universal_bools.append(bool2id[bool_var_name])

        for cls in clauses:
            self.bool_clauses.append([int(lit) for lit in cls.split(" ")])


def test_efsmt():
    from z3.z3util import get_vars
    x, y, z = z3.BitVecs("x y z", 6)
    fml = z3.Implies(z3.And(y > 0, y < 8), y - 2 * x < 7)

    universal_vars = [y]
    existential_vars = [item for item in get_vars(fml) if item not in universal_vars]
    m = EFBVFormulaManager()
    m.initialize(fml, existential_vars, universal_vars)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_efsmt()


