"""
FIXME: the file is very likely buggy
"""
import multiprocessing
from typing import List, Tuple

import z3

from arlib.utils.exceptions import ForAllSolverSuccess


def check_sat(fml: z3.ExprRef, fml_ctx, origin_ctx):
    solver = z3.Solver(ctx=fml_ctx)
    solver.add(fml)
    print(solver)
    if solver.check() == z3.sat:
        m = solver.model()
        return m.translate(origin_ctx)  # to the original context?
    else:
        raise ForAllSolverSuccess()


def parallel_check_sat_multiprocessing(fmls: List[z3.ExprRef], num_workers):
    """Solve clauses under a set of assumptions (deal with each one in parallel)
    FIXME: ValueError: ctypes objects containing pointers cannot be pickled
    """
    assert num_workers >= 1
    origin_ctx = fmls[0].ctx
    tasks = []
    for fml in fmls:
        # tasks.append((fml, main_ctx()))
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        tasks.append((i_fml, i_context))

    answers_async = [None for _ in fmls]
    with multiprocessing.Pool(num_workers) as p:
        def terminate_others(val):
            if val:
                p.terminate()  # TODO: when do we need this?

        for i, task in enumerate(tasks):
            answers_async[i] = p.apply_async(
                check_sat,
                (
                    task[0], task[1], origin_ctx
                ),
                callback=lambda val: terminate_others(val[0]))
        p.close()
        p.join()

    answers = [answer_async.get() for answer_async in answers_async if answer_async.ready()]
    res = [pres for pans, pres in answers]
    return res
