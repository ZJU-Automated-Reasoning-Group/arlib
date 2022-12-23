# coding: utf-8
import z3

from arlib.tests import TestCase, main
from arlib.tests.formula_generator import FormulaGenerator
from arlib.tests.grammar_gene import generate_from_grammar_as_str
from arlib.utils.z3_solver_utils import to_dnf


def is_sat(e):
    s = z3.Solver()
    s.add(e)
    s.set("timeout", 5000)
    return s.check() == z3.sat


class TestDNF(TestCase):

    def test_using_grammar_gene(self):
        try:
            fmlstr = generate_from_grammar_as_str(logic="QF_BV")
            if not fmlstr:  # generation error?
                return False
            fml = z3.And(z3.parse_smt2_string(fmlstr))
            if is_sat(fml):
                to_dnf(fml)
                return True
            return False
            # print(is_equivalent(qf, z3qf)) # TODO: use timeout
        except Exception:
            return False

    def test_using_api_gene(self):
        try:
            x, y, z = z3.BitVecs("x y z", 16)
            fg = FormulaGenerator([x, y, z])
            fml = fg.generate_formula()
            if is_sat(fml):
                to_dnf(fml)
                return True
            return False
        except Exception:
            return False


if __name__ == '__main__':
    main()
