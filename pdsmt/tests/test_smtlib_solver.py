# coding: utf-8
import logging

import z3

from . import TestCase, main
from .formula_generator import FormulaGenerator
from .grammar_gene import gene_smt2string
from ..global_params.paths import cvc5_exec, z3_exec
from ..theory import SMTLibTheorySolver, SMTLibPortfolioTheorySolver


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


class TestSMTLIBSolver(TestCase):

    def test_smtlib_solver(self):
        bin_cmd = cvc5_exec + " -q"
        bin_solver = SMTLibTheorySolver(bin_cmd)
        # smt2string = gen_small_formula("int")
        smt2string = gene_smt2string("QF_BV")
        bin_solver.add(smt2string)
        print(bin_solver.check_sat())

    def test_smtlib_portfolio_solver(self):
        return
        solvers_list = [cvc5_exec + " -q -i", z3_exec + " -in"]
        bin_solver = SMTLibPortfolioTheorySolver(solvers_list)
        bin_solver.add(gen_formula("real"))
        print(bin_solver.check_sat())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
