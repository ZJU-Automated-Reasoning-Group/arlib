# coding: utf-8
"""
For testing the quantifier elimination engine
"""

import z3

from arlib.tests import TestCase, main
from arlib.tests.formula_generator import FormulaGenerator
from arlib.quant.qe.qe_lme import qelim_exists_lme


def is_equivalent(a: z3.BoolRef, b: z3.BoolRef):
    """
    Check if a and b are equivalent
    """
    s = z3.Solver()
    s.set("timeout", 5000)
    s.add(a != b)
    if s.check() == z3.sat:
        return False
    return True


class TestQuantifierElimination(TestCase):

    def test_core_merge(self):
        import random
        w, x, y, z = z3.Ints("w x y z")
        fg = FormulaGenerator([w, x, y, z])
        fml = fg.generate_formula()
        qvars = random.choice([w, x, y, z])
        qf = qelim_exists_lme(fml, qvars)
        qfml = z3.Exists(qvars, fml)
        assert is_equivalent(qf, qfml)


if __name__ == '__main__':
    main()
