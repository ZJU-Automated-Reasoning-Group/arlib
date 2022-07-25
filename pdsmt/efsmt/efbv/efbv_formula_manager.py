# coding: utf-8
"""
For maintaining the correlations of bit-vec and Boolean world.
"""
import logging
from typing import List

import z3
from z3.z3util import get_vars

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

        logger.debug("existential vars: {}".format(self.existential_bools))
        logger.debug("universal vars:   {}".format(self.universal_bools))
        logger.debug("boolean clauses:  {}".format(self.bool_clauses))

    def to_qbf_clauses(self):
        """ Translate to a special QBF instance
        FIXME:
         After bit-blasting and CNF transformation, we may have many auxiliary Boolean variables.
         If operating over the Boolean level, it seems that we need to solve the problem below:
                 Exists BX ForAll BY Exists BZ . BF(BX, BY, BZ)  (where BZ is the set of auxiliary variables)
         Instead of the following problem
                 Exists X ForAll Y . F(X, Y)  (where X and Y are the existential and universal quantified bit-vectors, resp.)
        """
        prefix = "q"
        int2var = {}
        expr_clauses = []
        universal_vars = []
        existential_vars = []
        auxiliary_boolean_vars = []
        for clause in self.bool_clauses:
            expr_cls = []
            for numeric_lit in clause:
                # if numeric_lit == 0: break
                numeric_var = abs(numeric_lit)
                if numeric_var in int2var:
                    z3_var = int2var[numeric_var]
                else:
                    # create new Boolean vars
                    z3_var = z3.Bool("{0}{1}".format(prefix, numeric_var))
                    int2var[numeric_var] = z3_var
                    if numeric_var in self.universal_bools:
                        universal_vars.append(z3_var)
                    elif numeric_var in self.existential_bools:
                        existential_vars.append(z3_var)
                z3_lit = z3.Not(z3_var) if numeric_lit < 0 else z3_var
                expr_cls.append(z3_lit)
            expr_clauses.append(z3.Or(expr_cls))

        fml = z3.And(expr_clauses)
        # print("E vars: \n{}".format(existential_vars))
        # print("U vars: \n{}".format(universal_vars))
        # print("fml:    \n{}".format(fml))

        # { NOE: Ta trick for eliminating a subset of aux variables that are
        #     equivalent with existential or universal variables
        replace_mappings = []
        cared_vars_length = len(self.existential_bools) + len(self.universal_bools)

        for var_id in self.existential_bools:
            to_rep = z3.Bool("{0}{1}".format(prefix, var_id + cared_vars_length))
            var_z3 = z3.Bool("{0}{1}".format(prefix, var_id))
            replace_mappings.append((to_rep, var_z3))

        for var_id in self.universal_bools:
            to_rep = z3.Bool("{0}{1}".format(prefix, var_id + cared_vars_length))
            var_z3 = z3.Bool("{0}{1}".format(prefix, var_id))
            replace_mappings.append((to_rep, var_z3))
        # print("Rep mapping: \n {}".format(replace_mappings))
        # NOTE: the line below removes a subset of aux variables
        simplified_fml = z3.simplify(z3.substitute(fml, replace_mappings))
        # } End of the trick for eliminating a subset of aux variables.

        # the following loop collects the remaining aux variables
        for var in get_vars(simplified_fml):
            if not (var in universal_vars or var in existential_vars):
                auxiliary_boolean_vars.append(var)

        if len(auxiliary_boolean_vars) >= 1:
            cnt = z3.ForAll(universal_vars, z3.Exists(auxiliary_boolean_vars, simplified_fml))
        else:
            cnt = z3.ForAll(universal_vars, simplified_fml)
        return cnt


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
