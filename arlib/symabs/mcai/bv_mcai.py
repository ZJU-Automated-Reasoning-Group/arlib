"""
Model Counting Meets Abstract Interpretation

This module provides functionality to:
1. Count models using SharpSAT
2. Compute and analyze different abstract domains (Interval, Zone, Octagon)
3. Calculate false positive rates for each abstraction

Key Components:
- ModelCounter: Handles model counting operations
- AbstractionAnalyzer: Performs analysis across different abstract domains
- AbstractionResults: Stores false positive rates for each domain

Dependencies:
- z3: SMT solver for constraint solving
- arlib.smt.bv: Bit-vector model counting
- arlib.symabs: Symbolic abstraction implementations
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import z3

from arlib.smt.bv.qfbv_counting import BVModelCounter
from arlib.symabs.omt_symabs.bv_symbolic_abstraction import BVSymbolicAbstraction
from arlib.tests.formula_generator import FormulaGenerator

# from ..utils.plot_util import ScatterPlot  # See arlib/scripts


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AbstractionResults:
    """Store results from different abstraction domains"""
    interval_fp_rate: float = 0.0
    zone_fp_rate: float = 0.0
    octagon_fp_rate: float = 0.0


class ModelCounter:
    """Handles model counting operations"""

    def __init__(self, timeout_ms: int = 6000):
        self.timeout_ms = timeout_ms

    def is_sat(self, expression) -> bool:
        """Check if the expression is satisfiable"""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        solver.add(expression)
        return solver.check() == z3.sat

    def count_models(self, formula) -> int:
        """Count models using sharpSAT"""
        counter = BVModelCounter()
        counter.init_from_fml(formula)
        return counter.count_models_by_sharp_sat()


class AbstractionAnalyzer:
    """Analyzes different abstraction domains"""

    def __init__(self, formula, variables: List[z3.BitVecRef]):
        self.formula = formula
        self.variables = variables
        self.sa = BVSymbolicAbstraction()
        self.sa.init_from_fml(formula)

    def compute_false_positives(self, abs_formula) -> Tuple[bool, float]:
        """Compute false positive rate for an abstraction"""
        solver = z3.Solver()
        solver.add(z3.And(abs_formula, z3.Not(self.formula)))

        has_false_positives = solver.check() != z3.unsat
        if not has_false_positives:
            return False, 0.0

        # Count models for abstraction and false positives
        mc = BVModelCounter()
        mc.init_from_fml(abs_formula)
        abs_count = mc.count_models_by_sharp_sat()

        mc_fp = BVModelCounter()
        mc_fp.init_from_fml(z3.And(abs_formula, z3.Not(self.formula)))
        fp_count = mc_fp.count_models_by_sharp_sat()

        return True, fp_count / abs_count

    def analyze_abstractions(self) -> Optional[AbstractionResults]:
        """Analyze all abstraction domains"""
        try:
            # Perform abstractions
            self.sa.interval_abs()
            self.sa.zone_abs()
            self.sa.octagon_abs()

            results = AbstractionResults()

            # Analyze each domain
            for domain, formula in [
                ("Interval", self.sa.interval_abs_as_fml),
                ("Zone", self.sa.zone_abs_as_fml),
                ("Octagon", self.sa.octagon_abs_as_fml)
            ]:
                has_fp, fp_rate = self.compute_false_positives(formula)
                logger.info(f"{domain} domain: {'has FP rate %.4f' % fp_rate if has_fp else 'no false positives'}")

                if domain == "Interval":
                    results.interval_fp_rate = fp_rate
                elif domain == "Zone":
                    results.zone_fp_rate = fp_rate
                else:
                    results.octagon_fp_rate = fp_rate

            return results

        except Exception as e:
            logger.error(f"Error analyzing abstractions: {str(e)}")
            return None


def main():
    """Main entry point"""
    try:
        # Create test variables and formula
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]

        formula = FormulaGenerator(variables).generate_formula()
        counter = ModelCounter()

        if not counter.is_sat(formula):
            logger.info("Formula is unsatisfiable")
            return False

        # Count models
        model_count = counter.count_models(formula)
        logger.info(f"SharpSAT model count: {model_count}")

        # Analyze abstractions
        analyzer = AbstractionAnalyzer(formula, variables)
        results = analyzer.analyze_abstractions()

        if results:
            logger.info("Analysis completed successfully")
            return True

        return False

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return False


if __name__ == '__main__':
    main()
