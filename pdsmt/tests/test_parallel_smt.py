# coding: utf-8
import logging

import z3

from . import TestCase, main
from .formula_generator import FormulaGenerator
from ..cdcl.parallel_cdclt import parallel_cdclt


def gen_small_formula(logic: str):
    if logic == "int":
        w, x, y, z = z3.Ints("w x y z")
    elif logic == "real":
        w, x, y, z = z3.Reals("w x y z")
    elif logic == "bv":
        w, x, y, z = z3.BitVecs("w x y z", 8)
    else:
        w, x, y, z = z3.Reals("w x y z")
    fg = FormulaGenerator([x, y, z])
    fml = fg.generate_formula()
    s = z3.Solver()
    s.add(fml)
    return s.to_smt2()


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


class TestParallelSMTSolver(TestCase):

    def test_par_solver(self):
        logging.basicConfig(level=logging.DEBUG)

        for _ in range(10):
            # smt2string = gene_smt2string("QF_NRA")
            smt2string = gen_small_formula("real")
            try:
                fml = z3.And(z3.parse_smt2_string(smt2string))
                if is_simple_formula(fml):
                    continue
            except Exception as ex:
                print(ex)
                print(smt2string)
            res = parallel_cdclt(smt2string, logic="ALL")
            print(res)
            break  # exit when the first one is finished


if __name__ == '__main__':
    main()
