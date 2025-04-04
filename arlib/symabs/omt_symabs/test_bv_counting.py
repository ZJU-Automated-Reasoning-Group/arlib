# coding: utf-8

import z3
import itertools
import time
import math
import logging
import multiprocessing
from copy import deepcopy

from arlib.utils.z3_expr_utils import get_variables


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


def check_candidate_model(formula, all_vars, candidate):
    # The 1st approach, build a fake model
    # TODO: in some versions, ModelRef object has not attribute add_const_interp
    """
    m = Model()
    for i in range(len(self.vars)):
        m.add_const_interp(self.vars[i], BitVecVal(candidate[i], self.vars[i].sort().size()))
    if is_true(m.eval(self.formula), True): return True
    else: return False
    """
    solver = z3.Solver()
    solver.add(formula)
    # The 2nd approach, add them as additional constraints
    # for i in range(len(self.vars)):
    #    solver.add(self.vars[i] == BitVecVal(candidate[i], self.vars[i].sort().size()))
    # if solver.check() == sat: return True
    # else: return False
    # The 3rd approach, add them as assumption
    assumptions = []
    for i in range(len(all_vars)):
        assumptions.append(z3.BitVecVal(candidate[i], all_vars[i].sort().size()))
    if solver.check(assumptions) == z3.sat:
        return True
    else:
        return False


def check_candidate_models_set(formula, assignments, ctx):
    num_solutions = 0
    all_vars = get_variables(formula)
    for candidate in assignments:
        if check_candidate_model(formula, all_vars, candidate, ctx):
            num_solutions += 1
    print("num solutions a a subset: ", num_solutions)
    return num_solutions


def count_model_by_parallel_enumeration():
    # fvec = parse_smt2_file(fname)
    # formula = And(fvec)
    # vars = get_variables(formula)

    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    formula = z3.And(z3.UGT(x, 2), z3.UGT(y, 1))
    all_vars = get_variables(formula)

    time_start = time.process_time()
    logging.info("Start parallel BV enumeration-based")
    solutions = 0
    all_assignments = []  # can be very large
    for assignment in itertools.product(*[tuple(range(0, int(math.pow(2, x.sort().size())))) for x in all_vars]):
        all_assignments.append(assignment)

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    batched_assignments = split_list(all_assignments, cpus)
    results = []
    print("N cores: ", cpus)
    # https://github.com/Z3Prover/z3/blob/520ce9a5ee6079651580b6d83bc2db0f342b8a20/examples/python/parallel.py
    for i in range(0, cpus):
        # use new context
        i_context = z3.Context()
        formula_i = deepcopy(formula).translate(i_context)
        # TODO: this does not work
        result = pool.apply_async(check_candidate_models_set, args=(formula_i, batched_assignments[i], i_context))
        results.append(result)
    pool.close()
    pool.join()
    # final_res = []
    for result in results:
        print("on result: ", result)
        # final_res.append(result.get())
    print("Time:", time.process_time() - time_start)
    print("BV enumeration total solutions: ", solutions)
    return solutions


if __name__ == "__main__":
    count_model_by_parallel_enumeration()
