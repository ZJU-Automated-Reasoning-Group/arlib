"""Solving finite field formulas via bit-vector translation"""

from typing import Optional, Dict
import os

import z3
from .ff_parser import ParsedFormula, FieldExpr, FieldAdd, FieldMul, FieldEq, FieldVar, FieldConst


class FFBVSolver:
    """
    Solver for finite field formulas
    """

    def __init__(self, target_theory="QF_BV"):
        self.target_theory = target_theory
        self.solver = z3.Solver()
        self.sorts: Dict[str, z3.BitVecSort] = {}
        self.variables: Dict[str, z3.BitVecRef] = {}
        self.field_size: Optional[int] = None

    def translate_to_int(self, formula: ParsedFormula) -> z3.BoolRef:
        """
        Translate a finite field formula to an integer formula
        """
        raise NotImplementedError("Translation to int is not implemented.")

    def translate_to_bv(self, formula: ParsedFormula) -> z3.BoolRef:
        """
        Translate a finite field formula to a bit-vector formula
        """
        self.field_size = formula.field_size
        bits = (self.field_size - 1).bit_length()

        # Create sort
        sort = z3.BitVecSort(bits)

        # Translate variables
        for var_name, sort_name in formula.variables.items():
            var = z3.BitVec(var_name, bits)
            self.variables[var_name] = var
            # Add range constraint
            self.solver.add(z3.ULT(var, z3.BitVecVal(self.field_size, bits)))

        # Translate assertions
        for assertion in formula.assertions:
            self.solver.add(self._translate_expr(assertion))

        return self.solver.check()

    def _translate_expr(self, expr: FieldExpr) -> z3.ExprRef:
        if isinstance(expr, FieldAdd):
            result = self._translate_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = z3.URem(result + self._translate_expr(arg),
                                 z3.BitVecVal(self.field_size, result.size()))
            return result
        elif isinstance(expr, FieldMul):
            result = self._translate_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = z3.URem(result * self._translate_expr(arg),
                                 z3.BitVecVal(self.field_size, result.size()))
            return result
        elif isinstance(expr, FieldEq):
            return self._translate_expr(expr.left) == self._translate_expr(expr.right)
        elif isinstance(expr, FieldVar):
            return self.variables[expr.name]
        elif isinstance(expr, FieldConst):
            return z3.BitVecVal(expr.value, self.sorts[self.current_field].size())
        else:
            raise NotImplementedError(f"Translation not implemented for {type(expr)}")

    def get_model(self):
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None


def solve_qfff(smt_input):
    parser = FFParser()
    formula = parser.parse_formula(smt_input)
    solver = FFBVSolver()
    result = solver.translate_to_bv(formula)
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
