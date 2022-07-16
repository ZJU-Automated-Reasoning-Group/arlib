# coding: utf-8
"""
For maintaining the correlations of bit-vec and Boolean world.
"""
import logging
from typing import List

import z3

from ..bv import translate_smt2formula_to_cnf

logger = logging.getLogger(__name__)


def is_inconsistent(fml_a, fml_b):
    s = z3.Solver()
    s.add(z3.And(fml_a, fml_b))
    return s.check() == z3.unsat


class BVFormulaManager:

    def __init__(self):

        self.common_bool_vars_index = 1
        self.common_bool_vars_created = False
        self.common_bool_vars = []
        self.common_vars2bool = {}

    def mapped_bit_blast(self, fml: z3.BoolRef, cared_bv_vars: List[z3.ExprRef]):
        """"
        :param fml: a formula
        :param cared_bv_vars: the cared list of bit-vector variables (for ITP)
        :return: the corresponding cared list of Boolean variables and the blasted clauses
        """
        # print(fml)
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(fml)
        # print(bv2bool)
        # print(id_table)
        cared_bool_vars_numeric = []
        for bv_var in cared_bv_vars:
            # print(bv_var, ": corresponding bools")
            # print([id_table[bname] for bname in bv2bool[str(bv_var)]])
            if not self.common_bool_vars_created:
                self.common_vars2bool[bv_var] = []

            for bool_var_name in bv2bool[str(bv_var)]:
                cared_bool_vars_numeric.append(id_table[bool_var_name])

                if not self.common_bool_vars_created:
                    # create the common Boolean variables when translating fml_a!
                    # we should keep the mapping!
                    z3_var = z3.Bool("c{}".format(self.common_bool_vars_index))
                    self.common_bool_vars.append(z3_var)
                    self.common_bool_vars_index += 1

                    self.common_vars2bool[bv_var].append(z3_var)

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        # print(clauses_numeric)

        self.common_bool_vars_created = True  # a flag

        return cared_bool_vars_numeric, clauses_numeric

    def to_z3_clauses(self, prefix: str, cared_bool_vars: List[int], numeric_clauses: List[List[int]]):
        """
        :param prefix: to distinguish the fml_a and fml_b in interpolant generation
        :param cared_bool_vars: to label the common variables of fml_a and fml_b
        :param numeric_clauses: the corresponding numeric clauses of fml_a or fml_b
        :return:
        """
        int2var = {}
        expr_clauses = []
        for clause in numeric_clauses:
            expr_cls = []
            for numeric_lit in clause:
                # if numeric_lit == 0: break
                numeric_var = abs(numeric_lit)
                if numeric_var in int2var:
                    z3_var = int2var[numeric_var]
                else:
                    if numeric_var not in cared_bool_vars:
                        # create new Boolean vars
                        z3_var = z3.Bool("{0}{1}".format(prefix, numeric_var))
                        int2var[numeric_var] = z3_var
                    else:
                        # should look up self.common_bool_vars (created earlier)
                        var_index = cared_bool_vars.index(numeric_var)
                        z3_var = self.common_bool_vars[var_index]
                z3_lit = z3.Not(z3_var) if numeric_lit < 0 else z3_var
                expr_cls.append(z3_lit)
            expr_clauses.append(z3.Or(expr_cls))

        return z3.And(expr_clauses)

    def compute_itp(self, fml_a: z3.BoolRef, fml_b: z3.BoolRef, cared_vars: List[z3.ExprRef]):
        cared_bool_vars_a, clauses_a = self.mapped_bit_blast(fml_a, cared_vars)
        cared_bool_vars_b, clauses_b = self.mapped_bit_blast(fml_b, cared_vars)
        assert len(cared_bool_vars_a) == len(cared_bool_vars_b)
        # print("cared bool vars for fml_a and fml_b: ")
        # print(cared_bool_vars_a)
        # print(cared_bool_vars_b)

        z3_bool_fml_a = self.to_z3_clauses("a", cared_bool_vars_a, clauses_a)
        z3_bool_fml_b = self.to_z3_clauses("b", cared_bool_vars_b, clauses_b)

        # for debugging
        assert is_inconsistent(z3_bool_fml_a, z3_bool_fml_b)


def test_blast():
    x, y, z = z3.BitVecs("x y z", 3)
    # FXIME: why is this unsat?? (bugs in z3?)
    fml = z3.And(y == 1, x == 1, z < 4)
    s = z3.Solver()
    s.add(fml)
    print(s.check())
    translate_smt2formula_to_cnf(fml)
