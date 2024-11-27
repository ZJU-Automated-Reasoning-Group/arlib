import argparse
import logging
import multiprocessing as mp
# import csv
import random
import time
from typing import List, Tuple

import z3

from arlib.monabs.dis_check import disjunctive_check, disjunctive_check_incremental
from arlib.monabs.unary_check import unary_check, unary_check_cached
from arlib.tests.formula_generator import FormulaGenerator


def check_identical(*lists):
    mismatched_pairs = []

    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if lists[i] != lists[j]:
                mismatched_pairs.append((i, j))

    return mismatched_pairs


def run_single_test(logic_type: str, timeout: int) -> Tuple[str, List[int], float]:
    """Run a single test case with specified logic type and timeout"""
    if logic_type == "int":
        x, y, z = z3.Ints('x y z')
        init_vars = [x, y, z]
    elif logic_type == "real":
        x, y, z = z3.Reals('x y z')
        init_vars = [x, y,z]
    else:  # bitvector
        x, y, z = z3.BitVecs('x y z', 32)
        init_vars = [x, y, z]

    # Generate formulas
    generator = FormulaGenerator(init_vars)
    precond = generator.generate_formula()
    constraints = generator.get_preds(random.randint(5, 15))

    s = z3.Solver()
    s.set("timeout", 3000)
    s.add(precond)
    if s.check() != z3.sat:
        # the precond is unsat/unknown; no need to test
        return "invalid", -1

    # Measure execution time
    start_time = time.time()
    try:
        res_unary = unary_check(precond, constraints)
        res_unary_cached = unary_check_cached(precond, constraints)
        res_dis = disjunctive_check(precond, constraints)
        res_dis_inc = disjunctive_check_incremental(precond, constraints)

        # FIXME: has bugs...
        # res_con = intersection_based_check(precond, constraints)

        # check for inconsistency
        check_res = check_identical(res_unary, res_unary_cached,
                                    res_dis, res_dis_inc)

        if len(check_res) > 0:
            # find inconsistent results
            status = "failure"
        else:
            status = "success"
    except z3.Z3Exception as e:
        status = f"error: {str(e)}"
    execution_time = time.time() - start_time

    return status, execution_time


def worker(args: Tuple[int, str, int, str]) -> Tuple[int, str, List[int], float]:
    """Worker function for parallel execution"""
    test_id, logic_type, timeout, _ = args
    status, execution_time = run_single_test(logic_type, timeout)
    return test_id, status, execution_time


def main():
    # mp.cpu_count()
    parser = argparse.ArgumentParser(description='Z3 Formula Testing Script')
    parser.add_argument('--num_tests', type=int, default=100, help='Number of test cases')
    parser.add_argument('--logic_type', choices=['int', 'real', 'bv'], default='int', help='Logic type')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=1000, help='Timeout in milliseconds')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                        help='Logging level')
    parser.add_argument('--output', type=str, default='results.csv', help='Output CSV file')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Prepare test parameters
    test_params = [(i, args.logic_type, args.timeout, str(time.time()))
                   for i in range(args.num_tests)]

    # Run tests in parallel
    logger.info(f"Starting {args.num_tests} tests with {args.workers} workers")
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, test_params)

    # Write results to CSV
    """
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_id', 'logic_type', 'status', 'results', 'execution_time'])
        for test_id, status, test_results, exec_time in results:
            writer.writerow([test_id, args.logic_type, status, test_results, exec_time])

    logger.info(f"Results written to {args.output}")
    """

    # Print summary
    success_count = sum(1 for r in results if r[1] == "success")
    invalid_count = sum(1 for r in results if r[1] == "invalid")
    logger.info(f"Tests completed: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Invalid: {invalid_count}")
    logger.info(f"Failed: {len(results) - success_count - invalid_count}")


if __name__ == "__main__":
    main()
