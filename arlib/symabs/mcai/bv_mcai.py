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

import argparse
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing import Tuple

import z3
from z3 import parse_smt2_file

from arlib.counting.qfbv_counting import BVModelCounter
from arlib.symabs.omt_symabs.bv_symbolic_abstraction import BVSymbolicAbstraction
from arlib.tests.formula_generator import FormulaGenerator
from arlib.utils.z3_expr_utils import get_variables

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
    bitwise_fp_rate: float = 0.0


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
        self.formula = z3.And(formula)
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
            self.sa.bitwise_abs()

            results = AbstractionResults()

            # Analyze each domain
            for domain, formula in [
                ("Interval", self.sa.interval_abs_as_fml),
                ("Zone", self.sa.zone_abs_as_fml),
                ("Octagon", self.sa.octagon_abs_as_fml),
                ("Bitwise", self.sa.bitwise_abs_as_fml)
            ]:
                has_fp, fp_rate = self.compute_false_positives(formula)
                logger.info(f"{domain} domain: {'has FP rate %.4f' % fp_rate if has_fp else 'no false positives'}")

                if domain == "Interval":
                    results.interval_fp_rate = fp_rate
                elif domain == "Zone":
                    results.zone_fp_rate = fp_rate
                elif domain == "Octagon":
                    results.octagon_fp_rate = fp_rate
                elif domain == "Bitwise":
                    results.bitwise_fp_rate = fp_rate

            return results

        except Exception as e:
            logger.error(f"Error analyzing abstractions: {str(e)}, line {sys.exc_info()[-1].tb_lineno}")
            return None


def setup_logging(log_file: Optional[str] = None):
    """Configure logging to both file and console"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )
    return logging.getLogger(__name__)


def process_smt_file(file_path: str) -> bool:
    """Process a single SMT-LIB2 file"""
    try:
        # Parse SMT-LIB2 file
        formula = z3.And(parse_smt2_file(file_path))

        # Extract variables from formula
        # FIXME: is the following one correct?
        # variables = [var for var in formula.children()
        #              if var.sort().kind() == z3.Z3_BV_SORT]
        # import re

        # def extract_bv_variables(smt2_file):
        #     bv_vars = []
        #     pattern = r'\(declare-(?:const|fun)\s+(\S+)\s+(?:\(\s*\)\s+)?\(_ BitVec (\d+)\)\)'
        #     with open(smt2_file, 'r') as f:
        #         for line in f:
        #             line = line.strip()
        #             match = re.search(pattern, line)
        #             if match:
        #                 var_name = match.group(1)
        #                 width = int(match.group(2))
        #                 bv_vars.append(z3.BitVec(var_name, width))
        #     return bv_vars
        
        variables = get_variables(formula)
        # print(variables)

        if not variables:
            logger.warning(f"No bit-vector variables found in {file_path}")
            return False

        counter = ModelCounter()
        if not counter.is_sat(formula):
            logger.info(f"{file_path}: Formula is unsatisfiable")
            return False

        # Count models
        model_count = counter.count_models(formula)
        logger.info(f"{file_path}: SharpSAT model count: {model_count}")

        # Analyze abstractions
        analyzer = AbstractionAnalyzer(formula, variables)
        results = analyzer.analyze_abstractions()

        # TODO: also save the results in the log file (or some csv file)
        if results:
            logger.info(f"{file_path}: Analysis completed successfully")
            return True

        return False

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def process_directory(dir_path: str, num_processes: int) -> None:
    """Process all SMT-LIB2 files in directory using parallel processing"""
    smt_files = [
        str(f) for f in Path(dir_path).glob("**/*.smt2")
    ]

    if not smt_files:
        logger.warning(f"No SMT-LIB2 files found in {dir_path}")
        return

    logger.info(f"Found {len(smt_files)} SMT-LIB2 files to process")

    # TODO: also allow for sequential processing
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_smt_file, smt_files)

    successful = sum(1 for r in results if r)
    logger.info(f"Successfully processed {successful}/{len(smt_files)} files")


def main():
    """Main entry point with command line argument handling"""
    parser = argparse.ArgumentParser(
        description="Model counting and abstract interpretation analysis"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        help="Path to SMT-LIB2 file to analyze"
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Path to directory containing SMT-LIB2 files"
    )

    parser.add_argument(
        "-l", "--log",
        help="Path to log file (optional)",
        default=f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    parser.add_argument(
        "-p", "--processes",
        help="Number of parallel processes for directory processing",
        type=int,
        default=mp.cpu_count()
    )

    parser.add_argument(
        "-g", "--generate",
        help="Generate random formulas for demo",
        action='store_true'
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log)

    if args.generate:
        demo()
    else:
        try:
            if args.file:
                logger.info(f"Processing single file: {args.file}")
                success = process_smt_file(args.file)
                sys.exit(0 if success else 1)

            elif args.directory:
                logger.info(f"Processing directory: {args.directory}")
                process_directory(args.directory, args.processes)

        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            sys.exit(1)


def demo():
    try:
        # Create test variables and formula
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]

        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        logger.info(f"Generated formula: {sol.sexpr()}")
        counter = ModelCounter()

        while not counter.is_sat(formula):
            logger.info("Formula is unsatisfiable")
            formula = FormulaGenerator(variables).generate_formula()
            sol = z3.Solver()
            sol.add(formula)
            logger.info(f"Generated formula: {sol.sexpr()}")

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
    # demo()
