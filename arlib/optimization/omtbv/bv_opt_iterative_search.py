"""
TODO. pysmt cannot handle bit-vector operations in a few formulas generaed by z3.
 E.g., it may have a more strict restriction on
  the number of arguments to certain bit-vector options
"""
import logging

from arlib.utils import get_expr_vars

from arlib.optimization.pysmt_utils import *

logger = logging.getLogger(__name__)


def bv_opt_with_linear_search(z3_fml: z3.ExprRef, z3_obj: z3.ExprRef,
                              minimize: bool, solver_name: str):
    """Linear Search based OMT using PySMT with bit-vectors.
    solver_name: the backend SMT solver for pySMT
    """
    objname = z3_obj
    all_vars = get_expr_vars(z3_fml)
    if z3_obj not in all_vars:
        # NOTICE: we create a new variable to represent obj (a term, e.g., x + y)
        objname = z3.BitVec(str(z3_obj), z3_obj.sort().size())
        z3_fml = z3.And(z3_fml, objname == z3_obj)

    obj, fml = z3_to_pysmt(z3_fml, objname)
    # print(obj)
    # print(fml)
    logger.info("Starting linear search optimization")
    logger.debug(f"Optimization direction: {'minimize' if minimize else 'maximize'}")

    with Solver(name=solver_name) as solver:
        solver.add_assertion(fml)

        if minimize:
            lower = BV(0, obj.bv_width())
            iteration = 0
            while solver.solve():
                iteration += 1
                model = solver.get_model()
                lower = model.get_value(obj)
                solver.add_assertion(BVULT(obj, lower))
                logger.debug(f"Iteration {iteration}: Current lower bound = {lower}")
            logger.info(f"Minimization completed after {iteration} iterations")
            logger.info(f"Final minimum value: {lower}")
            return str(lower)
        else:
            cur_upper = None
            iteration = 0
            logger.info("Starting linear search maximization")
            while solver.solve():
                iteration += 1
                model = solver.get_model()
                cur_upper = model.get_value(obj)
                logger.debug(f"Iteration {iteration}: Current upper bound = {cur_upper}")
                solver.add_assertion(BVUGT(obj, cur_upper))

            result = str(cur_upper) if cur_upper is not None else "unsatisfiable"
            logger.info(f"Maximization completed after {iteration} iterations")
            logger.info(f"Final maximum value: {result}")
            return result


def bv_opt_with_binary_search(z3_fml, z3_obj, minimize: bool, solver_name: str):
    """Binary Search based OMT using PySMT with bit-vectors."""
    # Convert Z3 expressions to PySMT
    objname = z3_obj
    all_vars = get_expr_vars(z3_fml)
    if z3_obj not in all_vars:
        # NOTICE: we create a new variable to represent obj (a term, e.g., x + y)
        objname = z3.BitVec(str(z3_obj), z3_obj.sort().size())
        z3_fml = z3.And(z3_fml, objname == z3_obj)

    obj, fml = z3_to_pysmt(z3_fml, objname)

    # print(obj)
    # print(fml)

    sz = obj.bv_width()
    max_bv = (1 << sz) - 1
    logger.info("Starting binary search optimization")
    logger.debug(f"Optimization direction: {'minimize' if minimize else 'maximize'}")

    if not minimize:
        solver = Solver(name=solver_name)
        solver.add_assertion(fml)

        cur_min, cur_max = 0, max_bv
        upper = BV(0, sz)
        iteration = 0

        while cur_min <= cur_max:
            iteration += 1
            solver.push()

            cur_mid = cur_min + ((cur_max - cur_min) >> 1)
            logger.debug(f"Iteration {iteration}:")
            logger.debug(f"  min: {cur_min}, mid: {cur_mid}, max: {cur_max}")
            logger.debug(f"  current upper: {upper}")

            # cur_min_expr = BV(cur_min, sz)
            cur_mid_expr = BV(cur_mid, sz)
            cur_max_expr = BV(cur_max, sz)

            cond = And(BVUGE(obj, cur_mid_expr),
                       BVULE(obj, cur_max_expr))
            solver.add_assertion(cond)

            if not solver.solve():
                cur_max = cur_mid - 1
                logger.debug("  No solution found, reducing upper bound")
            else:
                model = solver.get_model()
                upper = model.get_value(obj)
                cur_min = int(upper.constant_value()) + 1
                logger.debug(f"  Found solution: {upper}")
            solver.pop()

        logger.info(f"Maximization completed after {iteration} iterations")
        logger.info(f"Final maximum value: {upper}")
        return upper
    else:
        # Compute minimum
        solver = Solver(name=solver_name)
        solver.add_assertion(fml)
        cur_min, cur_max = 0, max_bv
        lower = BV(max_bv, sz)
        iteration = 0

        while cur_min <= cur_max:
            iteration += 1
            solver.push()
            cur_mid = cur_min + ((cur_max - cur_min) >> 1)
            logger.debug(f"Iteration {iteration}:")
            logger.debug(f"  min: {cur_min}, mid: {cur_mid}, max: {cur_max}")
            logger.debug(f"  current lower: {lower}")

            cur_min_expr = BV(cur_min, sz)
            cur_mid_expr = BV(cur_mid, sz)
            # cur_max_expr = BV(cur_max, sz)
            cond = And(BVUGE(obj, cur_min_expr),
                       BVULE(obj, cur_mid_expr))
            solver.add_assertion(cond)

            if not solver.solve():
                cur_min = cur_mid + 1
                logger.debug("  No solution found, increasing lower bound")
            else:
                model = solver.get_model()
                lower = model.get_value(obj)
                cur_max = int(lower.constant_value()) - 1
                logger.debug(f"  Found solution: {lower}")
            solver.pop()

        min_value = lower
        logger.info(f"Minimization completed after {iteration} iterations")
        logger.info(f"Final minimum value: {lower}")
        return min_value


def demo_iterative():
    import time
    x, y, z = z3.BitVecs("x y z", 16)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    start_time = time.time()
    print("start solving")
    try:
        logger.info("\nRunning linear search maximization...")
        lin_res = bv_opt_with_linear_search(fml, y, minimize=False, solver_name="z3")
        logger.info(f"Linear search result: {lin_res}")

        logger.info("\nRunning binary search minimization...")
        bin_res = bv_opt_with_binary_search(fml, y, minimize=True, solver_name="z3")
        logger.info(f"Binary search result: {bin_res}")

        elapsed_time = time.time() - start_time
        logger.info(f"\nTotal solving time: {elapsed_time:.3f} seconds")
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")


def init_logger(log_level: str = 'INFO') -> None:
    """
    Initialize logger with specified logging level.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    # Clear any existing handlers
    logger.handlers.clear()

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Log initial message
    logger.debug(f"Logger initialized with level: {log_level}")


if __name__ == '__main__':
    # Set log level based on debug flag
    log_level = 'DEBUG'

    # Initialize logger
    init_logger(log_level)
    demo_iterative()
