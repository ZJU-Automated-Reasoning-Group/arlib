# coding: utf-8
"""
For testing the smtlib-based solver (used for interacting with binary solvers)
"""
import logging
from pathlib import Path

import z3

from arlib.tests import TestCase, main
from arlib.tests.formula_generator import FormulaGenerator
from arlib.tests.grammar_gene import gene_smt2string
from arlib.utils.smtlib_theory_solver import SMTLibTheorySolver, SMTLibPortfolioTheorySolver


def gen_small_formula(logic: str):
    """
    This function is for generating a small formula for testing
    """
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


project_root_dir = str(Path(__file__).parent.parent.parent)
z3_exec = project_root_dir + "/bin_solvers/z3"
cvc5_exec = project_root_dir + "/bin_solvers/cvc5"
print(cvc5_exec)


class TestSMTLIBSolver(TestCase):

    def test_smtlib_solver(self):
        bin_cmd = cvc5_exec + " -q"
        bin_solver = SMTLibTheorySolver(bin_cmd)
        # smt2string = gen_small_formula("int")
        smt2string = gene_smt2string("QF_BV")
        bin_solver.add(smt2string)
        print(bin_solver.check_sat())


if __name__ == '__main__':
    main()
