# coding: utf-8
"""
For testing CDCL(T)-based parallel SMT solving engine
"""

import logging

import z3

from arlib.tests import TestCase, main
from arlib.tests.formula_generator import FormulaGenerator
from arlib.efsmt.efbv.efbv_engine import solve_efsmt_bv
from arlib.efsmt.efbv.efbv_utils import EFBVResult


def gen_small_bv_formula(logic: str):
    assert logic == "bv"
    w, x, y = z3.BitVecs("w x y", 5)
    fg = FormulaGenerator([w, x, y])
    fml = fg.generate_formula()
    existential_vars = [w, y]
    universal_vars = [x]
    return existential_vars, universal_vars, fml


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


def solve_with_z3(universal_vars, fml):
    sol = z3.SolverFor("BV")
    sol.add(z3.ForAll(universal_vars, fml))
    res = sol.check()
    if res == z3.sat:
        return EFBVResult.SAT, sol.model()
    elif res == z3.unsat:
        return EFBVResult.UNSAT, None
    else:
        return EFBVResult.UNKNOWN, None


class TestEFBVSolver(TestCase):

    def test_efbv_solver(self):
        from z3.z3util import get_vars
        logging.basicConfig(level=logging.DEBUG)
        if True:
            for _ in range(15):
                existential_vars, universal_vars, fml = gen_small_bv_formula("bv")
                vars_fml = [str(v) for v in get_vars(fml)]
                if not ("w" in vars_fml and "x" in vars_fml and "y" in vars_fml):
                    continue
                if is_simple_formula(fml):
                    continue

                # print(fml)
                # print(existential_vars)

                res_a = solve_efsmt_bv(existential_vars, universal_vars, fml)
                res_b, model = solve_with_z3(universal_vars, fml)
                if res_a != res_b:
                    print("inconsistent!!")
                    print(res_a, res_b)
                    print(fml)
                    print(model)
                    break
                print(res_a, res_b)
                break


if __name__ == '__main__':
    main()
