"""
FIXME: the file is very likely buggy
"""
from multiprocessing import Pool
from typing import List, Tuple

import z3


def check_sat(fml: z3.ExprRef):
    print("Checking ..")
    sol = z3.SolverFor("QF_BV")
    sol.add(fml)
    print("Added to solver...")
    if sol.check() == z3.sat:
        m = sol.model()
        return m
    return None


def parallel_check_sat(fmls: List[z3.ExprRef], num_workers):
    """Solve clauses under a set of assumptions (deal with each one in parallel)
    TODO: Add timeout (if timeout, use the original model?)
    """
    assert num_workers >= 1
    answers_async = [None for _ in fmls]
    with Pool(num_workers) as p:
        def terminate_others(val):
            if val:
                p.terminate()  # TODO: when do we need this?

        for i, fml in enumerate(fmls):
            answers_async[i] = p.apply_async(
                check_sat,
                (
                    fml
                ),
                callback=lambda val: terminate_others(val[0]))
        p.close()
        p.join()

    answers = [answer_async.get() for answer_async in answers_async if answer_async.ready()]
    res = [pres for pans, pres in answers]
    return res
