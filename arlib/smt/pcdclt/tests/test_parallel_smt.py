# coding: utf-8
"""
For testing CDCL(T)-based parallel SMT solving engine
"""

import z3

from arlib.tests import TestCase, main
from arlib.tests.grammar_gene import gene_smt2string
from arlib.tests.misc import solve_with_z3
from arlib.smt.pcdclt.cdclt_solver import ParallelCDCLTSolver
from arlib.utils import SolverResult
from arlib.global_params import SMT_SOLVERS_PATH


def get_deterministic_formulas():
    """
    Returns a list of deterministic test formulas as SMT-LIB2 strings
    along with their expected results.
    """
    formulas = []

    # Formula 1: Simple satisfiable linear real arithmetic
    formula1 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (assert (> x 0))
    (assert (< y 10))
    (assert (>= (+ x y) 5))
    """
    formulas.append((formula1, SolverResult.SAT, "Simple satisfiable LRA"))

    # Formula 2: Simple unsatisfiable linear real arithmetic - make sure it's properly formatted
    formula2 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (assert (and (> x 0) (< x 0)))
    """
    formulas.append((formula2, SolverResult.UNSAT, "Simple unsatisfiable LRA"))

    # Formula 3: More complex satisfiable formula
    formula3 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (declare-fun z () Real)
    (assert (>= x 0))
    (assert (<= y 10))
    (assert (>= z 2))
    (assert (<= z 8))
    (assert (or (>= (+ x y) 15) (and (>= x 5) (<= y 3))))
    """
    formulas.append((formula3, SolverResult.SAT, "Complex satisfiable LRA"))

    # Formula 4: More complex unsatisfiable formula
    formula4 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (declare-fun z () Real)
    (assert (and (>= x 0) (<= x 5)))
    (assert (and (>= y 0) (<= y 5)))
    (assert (and (>= z 0) (<= z 5)))
    (assert (>= (+ x y z) 20))
    """
    formulas.append((formula4, SolverResult.UNSAT, "Complex unsatisfiable LRA"))

    # Formula 5: Linear real arithmetic with many variables
    formula5 = """
    (set-logic QF_LRA)
    (declare-fun a () Real)
    (declare-fun b () Real)
    (declare-fun c () Real)
    (declare-fun d () Real)
    (declare-fun e () Real)
    (assert (>= a 0))
    (assert (>= b 0))
    (assert (>= c 0))
    (assert (>= d 0))
    (assert (>= e 0))
    (assert (<= (+ a b c d e) 10))
    (assert (>= (+ a b) 3))
    (assert (>= (+ c d) 4))
    (assert (>= e 2))
    """
    formulas.append((formula5, SolverResult.SAT, "LRA with many variables"))

    # Formula 6: Boolean combinatorics
    formula6 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (assert (or (and (> x 0) (< y 0))
                (and (< x 0) (> y 0))))
    (assert (or (and (> x 1) (> y -1))
                (and (< x -1) (< y -2))))
    """
    formulas.append((formula6, SolverResult.SAT, "Boolean combinatorics"))

    # Formula 7: System of equations (satisfiable)
    formula7 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (declare-fun z () Real)
    (assert (= (+ x y) 10))
    (assert (= (+ y z) 20))
    (assert (= (+ x z) 15))
    """
    formulas.append((formula7, SolverResult.SAT, "Satisfiable system of equations"))

    # Formula 8: Truly unsatisfiable system of equations
    formula8 = """
    (set-logic QF_LRA)
    (declare-fun x () Real)
    (declare-fun y () Real)
    (declare-fun z () Real)
    (assert (= x 5))
    (assert (= y 10))
    (assert (= z 20))
    (assert (= (+ x y z) 10))  ; 5 + 10 + 20 = 35, not 10
    """
    formulas.append((formula8, SolverResult.UNSAT, "Truly unsatisfiable system of equations"))

    return formulas


class TestParallelSMTSolver(TestCase):


    def test_deterministic_formulas(self):
        """Test with deterministic formulas for reproducible results"""

        # First check if the Z3 path is correctly set with the "-in" argument
        z3_config = SMT_SOLVERS_PATH.get('z3', {})
        if not z3_config.get('available', False) or "-in" not in z3_config.get('args', ""):
            self.skipTest("Z3 not available or not configured with -in argument")

        for formula, expected_result, formula_name in get_deterministic_formulas():
            print(f"\nTesting formula: {formula_name}")

            # First check with Z3 to confirm our expected result
            z3_result = solve_with_z3(formula)
            print(f"  Z3 result: {z3_result}, Expected: {expected_result}")

            # Ensure our reference is correct
            self.assertEqual(z3_result, expected_result,
                            f"Z3 reference gave {z3_result} but expected {expected_result} for {formula_name}")

            # Then test with our parallel solver
            sol = ParallelCDCLTSolver(mode="process")
            result = sol.solve_smt2_string(formula, logic="ALL")
            print(f"  Parallel solver result: {result}")

            # Check the parallel solver matches the expected result
            self.assertEqual(result, expected_result,
                            f"Parallel solver gave {result} but expected {expected_result} for {formula_name}")

    def test_par_solver(self):
        """
        Test the parallel solver with a randomly generated formula
        """

        # First check if the Z3 path is correctly set with the "-in" argument
        z3_config = SMT_SOLVERS_PATH.get('z3', {})
        if not z3_config.get('available', False) or "-in" not in z3_config.get('args', ""):
            self.skipTest("Z3 not available or not configured with -in argument")

        # Generate a random SMT formula in QF_LRA logic
        smt2string = gene_smt2string("QF_LRA")

        # Skip test if the output is an error message or usage information
        if isinstance(smt2string, str) and (smt2string.startswith('usage:') or 'error:' in smt2string):
            self.skipTest("Invalid SMT-LIB2 formula generated")

        try:
            # Check if the formula is parseable
            z3.parse_smt2_string(smt2string)
        except Exception as ex:
            print(ex)
            print(smt2string)
            self.skipTest(f"Error parsing SMT2 string: {ex}")

        # Solve with our parallel solver
        sol = ParallelCDCLTSolver(mode="process")
        res = sol.solve_smt2_string(smt2string, logic="ALL")

        # Solve with Z3 for reference
        res_z3 = solve_with_z3(smt2string)

        print(f"Parallel solver result: {res}, Z3 result: {res_z3}")

        # Check that both solvers agree
        self.assertEqual(res, res_z3,
                        f"Inconsistent results: Parallel solver gave {res} but Z3 gave {res_z3}")


if __name__ == '__main__':
    main()
