#!/usr/bin/env python3
"""
Tests for the SMT formula solver.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import z3

from qfbv_portfolio import SolverResult, SATSolver, FormulaParser, SolverConfig

class TestSolverResult(unittest.TestCase):
    """Test SolverResult enum functionality."""

    def test_return_codes(self):
        """Test return code mappings."""
        self.assertEqual(SolverResult.SAT.return_code, 10)
        self.assertEqual(SolverResult.UNSAT.return_code, 20)
        self.assertEqual(SolverResult.UNKNOWN.return_code, 0)

    def test_value_conversion(self):
        """Test string value conversion."""
        self.assertEqual(SolverResult("sat"), SolverResult.SAT)
        self.assertEqual(SolverResult("unsat"), SolverResult.UNSAT)
        self.assertEqual(SolverResult("unknown"), SolverResult.UNKNOWN)

class TestSATSolver(unittest.TestCase):
    """Test SAT solver functionality."""

    @staticmethod
    def create_sample_cnf():
        """Creates a simple satisfiable CNF formula."""
        from pysat.formula import CNF
        cnf = CNF()
        # (x1 ∨ x2) ∧ (¬x1 ∨ x2)
        cnf.append([1, 2])
        cnf.append([-1, 2])
        return cnf

    def test_solve_sat(self):
        """Test basic SAT solving."""
        import multiprocessing
        cnf = self.create_sample_cnf()
        queue = multiprocessing.Queue()
        
        # Test with different solvers
        for solver_name in ['g4', 'mc']:  # Testing with just two solvers for speed
            SATSolver.solve_sat(solver_name, cnf, queue)
            result = queue.get()
            self.assertEqual(result, SolverResult.SAT)

class TestFormulaParser(unittest.TestCase):
    """Test SMT formula parsing and solving."""

    @classmethod
    def setUpClass(cls):
        """Set up test cases with temporary files."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_cases = {
            "sat_case": "(declare-const x Bool) (assert (= x x))",
            "unsat_case": "(declare-const x Bool) (assert (and x (not x)))",
            "simple_bv": """
                (declare-const a (_ BitVec 4))
                (declare-const b (_ BitVec 4))
                (assert (= a b))
                (assert (= a #x5))
                (assert (= b #x6))
            """
        }
        
        # Create test SMT2 files
        cls.test_files = {}
        for name, content in cls.test_cases.items():
            file_path = os.path.join(cls.temp_dir, f"{name}.smt2")
            with open(file_path, "w") as f:
                f.write(content)
            cls.test_files[name] = file_path

    def test_sat_formula(self):
        """Test satisfiable formula."""
        result = FormulaParser.solve(self.test_files["sat_case"])
        # print(result)
        self.assertEqual(result, SolverResult.SAT)

    def test_unsat_formula(self):
        """Test unsatisfiable formula."""
        result = FormulaParser.solve(self.test_files["unsat_case"])
        self.assertEqual(result, SolverResult.UNSAT)

    def test_bv_formula(self):
        """Test bit-vector formula."""
        result = FormulaParser.solve(self.test_files["simple_bv"])
        self.assertEqual(result, SolverResult.UNSAT)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        for file_path in cls.test_files.values():
            try:
                os.remove(file_path)
            except OSError:
                pass
        try:
            os.rmdir(cls.temp_dir)
        except OSError:
            pass

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete solver pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test input.json
        self.input_json = {
            "formula_file": os.path.join(self.temp_dir, "test.smt2")
        }
        with open(os.path.join(self.temp_dir, "input.json"), "w") as f:
            json.dump(self.input_json, f)

    def create_test_case(self, formula: str) -> Dict:
        """Helper to create and run test cases."""
        # Write formula to SMT2 file
        with open(self.input_json["formula_file"], "w") as f:
            f.write(formula)
        
        # Run solver
        result = FormulaParser.solve(self.input_json["formula_file"])
        return {
            "result": result,
            "return_code": result.return_code
        }

    def test_complete_pipeline(self):
        """Test the complete solving pipeline with different formulas."""
        test_cases = [
            {
                "formula": "(declare-const x Bool) (assert (= x x))",
                "expected_result": SolverResult.SAT,
                "expected_code": 10
            },
            {
                "formula": "(declare-const x Bool) (assert (and x (not x)))",
                "expected_result": SolverResult.UNSAT,
                "expected_code": 20
            },
            {
                "formula": """
                    (declare-const a (_ BitVec 8))
                    (declare-const b (_ BitVec 8))
                    (assert (bvult a b))
                    (assert (= a #xff))
                """,
                "expected_result": SolverResult.UNSAT,
                "expected_code": 20
            }
        ]

        for case in test_cases:
            result = self.create_test_case(case["formula"])
            self.assertEqual(result["result"], case["expected_result"])
            self.assertEqual(result["return_code"], case["expected_code"])

    def tearDown(self):
        """Clean up temporary files."""
        try:
            os.remove(self.input_json["formula_file"])
            os.remove(os.path.join(self.temp_dir, "input.json"))
            os.rmdir(self.temp_dir)
        except OSError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
    