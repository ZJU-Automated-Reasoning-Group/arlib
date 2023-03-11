"""Model counting for QF_BV formulas
"""

from typing import List
import itertools
import logging
import math
import multiprocessing
from copy import deepcopy
from timeit import default_timer as counting_timer

import z3
from arlib.utils.z3_expr_utils import get_variables
from arlib.bv.mapped_blast import translate_smt2formula_to_cnf
from arlib.bool.counting.dimacs_counting import count_dimacs_solutions


def split_list(alist, wanted_parts=1):
    if wanted_parts == 0:
        raise ZeroDivisionError("wanted_parts must be greater than zero.")
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


def check_candidate_model(formula, all_vars, candidate):
    """The 1st approach, build a fake model
     TODO: in some versions, ModelRef object has not attribute add_const_interp
    m = Model()
    for i in range(len(self.vars)):
        m.add_const_interp(self.vars[i], BitVecVal(candidate[i], self.vars[i].sort().size()))
    if is_true(m.eval(self.formula), True): return True
    else: return False
    """
    solver = z3.Solver(ctx=formula.ctx)
    solver.add(formula)
    assumption = []
    for i in range(len(all_vars)):
        assumption.append(all_vars[i] == z3.BitVecVal(candidate[i], all_vars[i].sort().size()))
    if solver.check(assumption) == z3.sat:
        return True
    else:
        return False


def check_candidate_models_set(formula: z3.ExprRef, assignments: List):
    num_solutions = 0
    variables = get_variables(formula)
    for candidate in assignments:
        if check_candidate_model(formula, variables, candidate):
            num_solutions += 1
    print("num solutions a a subset: ", num_solutions)
    return num_solutions


class BVModelCounter:
    """
        A class for counting the number of models of a Z3 formula.
    Attributes:
        formula (z3.ExprRef): The Z3 formula to count models for.
        vars (list): A list of all variables in the formula.
    """

    def __init__(self):
        self.formula = None
        self.vars = []
        self.counts = 0
        self.smt2file = None

    def init_from_file(self, filename):
        try:
            self.smt2file = filename
            self.formula = z3.And(z3.parse_smt2_file(filename))
            for var in get_variables(self.formula):
                if z3.is_bv(var):
                    # print(var)
                    self.vars.append(var)
            logging.debug("Init model counter success!")
        except z3.Z3Exception as ex:
            print(ex)
            return None

    def init_from_fml(self, fml):
        try:
            self.formula = fml
            for var in get_variables(self.formula):
                if z3.is_bv(var):
                    self.vars.append(var)
        except z3.Z3Exception as ex:
            print(ex)
            return None

    def count_model_by_bv_enumeration(self):
        """
        Enumerate all possible assignments
        TODO: handle signed and the mixing of signed and unsigned
        """
        # time_start = time.process_time()
        time_start = counting_timer()
        logging.debug("Start BV enumeration-based")
        solutions = 0
        for assignment in itertools.product(*[tuple(range(0, int(math.pow(2, x.sort().size())))) for x in self.vars]):
            # print(assignment)
            ret = check_candidate_model(self.formula, self.vars, assignment)
            if ret:
                solutions = solutions + 1
        print("Time:", counting_timer() - time_start)
        print("BV enumeration total solutions: ", solutions)
        return solutions

    # TODO: fix
    def count_model_by_enumeration_parallel(self):
        # time_start = time.process_time()
        time_start = counting_timer()
        logging.debug("Start parallel BV enumeration-based")
        solutions = 0
        all_assignments = []  # can be very large
        for assignment in itertools.product(*[tuple(range(0, int(math.pow(2, x.sort().size())))) for x in self.vars]):
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
            formula_i = deepcopy(self.formula).translate(i_context)
            # TODO: this does not work
            result = pool.apply_async(check_candidate_models_set, args=(formula_i, batched_assignments[i], i_context))
            results.append(result)
        pool.close()
        pool.join()
        final_res = []
        for result in results:
            print("on result: ", result)
            final_res.append(result.get())
        # TODO: check in parallel
        print("Time:", counting_timer() - time_start)
        print("BV enumeration total solutions: ", solutions)
        return solutions

    def count_models_by_sharp_sat(self):
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.formula)
        time_start = counting_timer()
        solutions = count_dimacs_solutions(header, clauses)
        print("Time:", counting_timer() - time_start)
        print("sharpSAT total solutions: ", solutions)
        return solutions


def feat_test():
    mc = BVModelCounter()
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
    mc.init_from_fml(fml)
    # mc.init_from_file('../../benchmarks/t1.smt2')
    # mc.count_model_by_bv_enumeration()
    mc.count_models_by_sharp_sat()


if __name__ == '__main__':
    feat_test()
