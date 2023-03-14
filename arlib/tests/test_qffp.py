# coding: utf-8
"""
For testing the QF_FP solver
"""

from arlib.tests import TestCase, main
from arlib.tests.grammar_gene import gene_smt2string
from arlib.fp.qffp_solver import QFFPSolver
from arlib.utils import SolverResult

import z3


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then('simplify', 'propagate-values')(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


def solve_with_z3(smt2string: str):
    fml = z3.And(z3.parse_smt2_string(smt2string))
    sol = z3.Solver()
    sol.add(fml)
    res = sol.check()
    if res == z3.sat:
        return SolverResult.SAT
    elif res == z3.unsat:
        return SolverResult.UNSAT
    else:
        return SolverResult.UNKNOWN


class TestQFFP(TestCase):
    """
    Test the bit-blasting based solver
    """

    def test_qffp_solver(self):
        i = 0
        for _ in range(10):

            smt2string = gene_smt2string("QF_FP")
            try:
                fml = z3.And(z3.parse_smt2_string(smt2string))
                if is_simple_formula(fml):
                    continue
            except Exception as ex:
                print(ex)
                print(smt2string)

            i = i + 1
            print("!!!Solving {}-th formula!!!".format(i))

            # sol = ParallelCDCLSolver(mode="process")
            sol = QFFPSolver()
            res = sol.solve_smt_string(smt2string)
            res_z3 = solve_with_z3(smt2string)
            print(res, res_z3)
            if res != res_z3:
                print("inconsistent!!")

            # break  # exit when the first one is finished


if __name__ == '__main__':
    main()
