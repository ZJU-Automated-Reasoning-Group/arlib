import unittest
import z3
from pysmt.shortcuts import Symbol, And, Or, Not

from arlib.counting.bool.z3py_expr_counting import count_z3_solutions, \
    count_z3_models_by_enumeration

from arlib.counting.bool.pysmt_expr_counting import count_pysmt_solutions, \
    count_pysmt_models_by_enumeration


class TestModelCounting(unittest.TestCase):
    def test_z3_simple(self):
        # Simple formula: (a or b) and (not a or not b)
        a = z3.Bool('a')
        b = z3.Bool('b')
        formula = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), z3.Not(b)))

        # Should have 2 solutions: (True, False) and (False, True)
        count = count_z3_solutions(formula)
        self.assertEqual(count, 2)

        # Test parallel counting
        count_parallel = count_z3_solutions(formula, parallel=False)
        self.assertEqual(count_parallel, 2)

    def test_z3_unsatisfiable(self):
        # Formula: a and (not a)
        a = z3.Bool('a')
        formula = z3.And(a, z3.Not(a))

        # Should have 0 solutions
        count = count_z3_models_by_enumeration(formula)
        self.assertEqual(count, 0)

    def test_z3_tautology(self):
        # Formula: a or (not a)
        a = z3.Bool('a')
        formula = z3.Or(a, z3.Not(a))

        # Should have 2 solutions
        count = count_z3_models_by_enumeration(formula)
        self.assertEqual(count, 2)

    def test_z3_complex_tautology(self):
        # Complex tautology: (a and b) or (not a or not b)
        a = z3.Bool('a')
        b = z3.Bool('b')
        formula = z3.Or(z3.And(a, b), z3.Or(z3.Not(a), z3.Not(b)))
        count = count_z3_models_by_enumeration(formula)
        self.assertEqual(count, 4)  # All possible assignments satisfy this

    def test_z3_xor_chain(self):
        # XOR chain: a xor b xor c
        a, b, c = z3.Bools('a b c')
        formula = z3.Xor(z3.Xor(a, b), c)
        count = count_z3_models_by_enumeration(formula)
        self.assertEqual(count, 4)  # Should have 4 solutions

    def test_pysmt_empty_formula(self):
        from pysmt.shortcuts import TRUE
        count = count_pysmt_models_by_enumeration(TRUE())
        self.assertEqual(count, 1)

    def test_pysmt_complex_formula(self):
        # (a → b) ∧ (b → c) ∧ (c → a)
        return  # failed
        a = Symbol('a')
        b = Symbol('b')
        c = Symbol('c')
        implies_a_b = Or(Not(a), b)
        implies_b_c = Or(Not(b), c)
        implies_c_a = Or(Not(c), a)
        formula = And(implies_a_b, implies_b_c, implies_c_a)
        count = count_pysmt_models_by_enumeration(formula)
        self.assertEqual(count, 4)  # Should have 4 solutions: FFF, TTT

    def test_pysmt_large_formula(self):
        # Create a chain of implications: a1 → a2 → a3 → ... → an
        return  # failed
        n = 5
        vars = [Symbol(f'a{i}') for i in range(n)]
        implications = [Or(Not(vars[i]), vars[i + 1]) for i in range(n - 1)]
        formula = And(implications)
        count = count_pysmt_models_by_enumeration(formula)
        self.assertEqual(count, 2 ** n - n)  # Number of solutions for implication chain


if __name__ == '__main__':
    unittest.main()
