"""Solving finite field formulas via integer translation"""

from typing import Optional, Dict
import os

import z3
from arlib.smt.ff.ff_parser import ParsedFormula, FieldExpr, FieldAdd, FieldMul, FieldEq, FieldVar, FieldConst, FFParser


class FFIntSolver:
    """
    Solver for finite field formulas
    """

    def __init__(self, target_theory="QF_NIA"):
        self.target_theory = target_theory
        self.solver = z3.Solver()
        self.variables: Dict[str, z3.IntRef] = {}
        self.field_size: Optional[int] = None

    def translate_to_int(self, formula: ParsedFormula) -> z3.BoolRef:
        """
        Translate a finite field formula to an integer formula
        """
        self.field_size = formula.field_size

        # Translate variables
        for var_name, sort_name in formula.variables.items():
            var = z3.Int(var_name)
            self.variables[var_name] = var
            # Add range constraint: 0 <= var < field_size
            self.solver.add(z3.And(var >= 0, var < self.field_size))

        # Translate assertions
        for assertion in formula.assertions:
            self.solver.add(self._translate_expr(assertion))

        return self.solver.check()

    def _translate_expr(self, expr: FieldExpr) -> z3.ExprRef:
        if isinstance(expr, FieldAdd):
            result = self._translate_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = (result + self._translate_expr(arg)) % self.field_size
            return result
        elif isinstance(expr, FieldMul):
            result = self._translate_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = (result * self._translate_expr(arg)) % self.field_size
            return result
        elif isinstance(expr, FieldEq):
            return self._translate_expr(expr.left) == self._translate_expr(expr.right)
        elif isinstance(expr, FieldVar):
            return self.variables[expr.name]
        elif isinstance(expr, FieldConst):
            return expr.value
        else:
            raise NotImplementedError(f"Translation not implemented for {type(expr)}")

    def get_model(self):
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None


def solve_qfff(smt_input):
    parser = FFParser()
    formula = parser.parse_formula(smt_input)
    solver = FFIntSolver()
    result = solver.translate_to_int(formula)
    if result == z3.sat:
        model = solver.get_model()
        print("Satisfiable")
    elif result == z3.unsat:
        print("Unsatisfiable")
    else:
        print("Unknown")


def regress(dir: str):
    """Run regression tests on all SMT2 files in directory."""
    for filename in os.listdir(dir):
        if filename.endswith(".smt2"):
            with open(os.path.join(dir, filename), 'r') as file:
                smt_input = file.read()
                print(f"Testing {filename}...")

                # Get expected result
                if "(set-info :status 'sat')" in smt_input:
                    expected = "Satisfiable"
                elif "(set-info :status 'unsat')" in smt_input:
                    expected = "Unsatisfiable"
                else:
                    expected = "Unknown"

                print(f"Expected: {expected}")
                solve_qfff(smt_input)


def demo():
    """Demonstration of the finite field solver."""
    smt_input = """
(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(set-logic QF_FF)
(declare-fun x () (_ FiniteField 17))
(declare-fun m () (_ FiniteField 17))
(declare-fun is_zero () (_ FiniteField 17))
(assert (not (=>
  (and (= #f0m17 (ff.add (ff.mul m x) #f16m17 is_zero))
       (= #f0m17 (ff.mul is_zero x)))
  (and (or (= #f0m17 is_zero) (= #f1m17 is_zero))
       (= (= #f1m17 is_zero) (= x #f0m17)))
)))
(check-sat)
    """
    solve_qfff(smt_input)


if __name__ == '__main__':
    # demo()
    from pathlib import Path

    current_file = Path(__file__)
    ff_dir = current_file.parent.parent.parent.parent / "benchmarks" / "smtlib2" / "ff"
    regress(str(ff_dir))
