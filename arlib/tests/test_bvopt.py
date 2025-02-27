# coding: utf-8
"""
For testing OMT(BV) solving engine
"""
import z3

from arlib.tests import TestCase, main
from arlib.optimization.bvopt.qfbv_opt import BitBlastOMTBVSolver


class TestBVOMT(TestCase):
    """Test the OMT(BV) solver with deterministic test cases"""

    def setUp(self):
        self.solver = BitBlastOMTBVSolver()

    def test_simple_maximization(self):
        """Test simple maximization with basic constraints"""
        x = z3.BitVec('x', 8)
        y = z3.BitVec('y', 8)
        
        # Test case: x < 100 && x + y == 20
        formula = z3.And(z3.ULT(x, z3.BitVecVal(100, 8)), 
                        x + y == z3.BitVecVal(20, 8))
        
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize(x, is_signed=False)
        self.assertIsNotNone(result)
        self.assertTrue(result < 100)

    def test_signed_maximization(self):
        """Test maximization with signed bit-vectors"""
        # FIXME: is this supported?
        return
        x = z3.BitVec('x', 8)
        y = z3.BitVec('y', 8)
        
        # Test case: x > -50 && x < 50 && x + y == 10
        formula = z3.And(
            x > z3.BitVecVal(-50, 8),
            x < z3.BitVecVal(50, 8),
            x + y == z3.BitVecVal(10, 8)
        )
        
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize(x, is_signed=True)
        self.assertIsNotNone(result)
        self.assertTrue(-50 < result < 50)

    def test_multiple_constraints(self):
        """Test optimization with multiple constraints"""
        x, y, z = z3.BitVecs('x y z', 8)
        
        formula = z3.And(
            z3.ULT(x + y, z3.BitVecVal(200, 8)),
            z3.UGT(x, z3.BitVecVal(50, 8)),
            x + y == z,
            z3.ULT(z, z3.BitVecVal(150, 8))
        )
        
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize(x, is_signed=False)
        self.assertIsNotNone(result)
        # self.assertTrue(50 < result < 150)

    def test_different_engines(self):
        """Test different optimization engines"""
        x, y = z3.BitVecs('x y', 8)
        formula = z3.And(
            z3.ULT(x, z3.BitVecVal(100, 8)),
            x + y == z3.BitVecVal(50, 8)
        )
        
        self.solver.from_smt_formula(formula)
        
        # Test MaxSAT-based optimization
        result1 = self.solver.maximize_with_maxsat(x, is_signed=False)
        self.assertIsNotNone(result1)
        
        # TODO: Add tests for other engines when implemented:
        # - Quantifier-based optimization
        # - Binary search optimization
        # - Linear search optimization

    def test_boundary_cases(self):
        """Test optimization with boundary cases"""
        x = z3.BitVec('x', 8)
        
        # Test maximum possible value
        formula = z3.ULT(x, z3.BitVecVal(255, 8))
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize_with_maxsat(x, is_signed=False)
        self.assertEqual(result, 254)
        
        # Test minimum possible value
        formula = z3.UGT(x, z3.BitVecVal(0, 8))
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize_with_maxsat(x, is_signed=False)
        self.assertEqual(result, 255)
        
        # Test with exact value constraint
        formula = x == z3.BitVecVal(128, 8)
        self.solver.from_smt_formula(formula)
        result = self.solver.maximize_with_maxsat(x, is_signed=False)
        self.assertEqual(result, 128)


if __name__ == '__main__':
    main()
