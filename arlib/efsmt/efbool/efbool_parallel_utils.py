from multiprocessing import Pool
from typing import List, Tuple

from pysat.solvers import Solver


# from pysat.formula import CNF


def check_sat_assuming(clauses: List[List[int]], assumptions: List[int]) -> Tuple:
    """Check satisfiability of a formula under som assumption
    :param clauses: a Boolean formula
    :param assumptions: a set of literals (forming the assumptions)
    :return: (true, model) or (false, unsat core)?
    """
    solver = Solver(name="m22", bootstrap_with=clauses)
    ans = solver.solve(assumptions=assumptions)
    if ans:
        return ans, solver.get_model()
    return ans, solver.get_core()


def parallel_check_assumptions(clauses: List[List[int]], assumptions_lists: List[List[int]], num_workers: int):
    """Solve clauses under a set of assumptions (deal with each one in parallel)
    TODO: - Should we enforce that clauses are satisfiable?
          - Add timeout (if timeout, use the original model?)
    """
    assert num_workers >= 1
    answers_async = [None for _ in assumptions_lists]
    with Pool(num_workers) as p:
        def terminate_others(val):
            if val:
                p.terminate()  # TODO: when do we need this?

        for i, assumptions in enumerate(assumptions_lists):
            answers_async[i] = p.apply_async(
                check_sat_assuming,
                (
                    clauses,
                    assumptions
                ),
                callback=lambda val: terminate_others(val[0]))
        p.close()
        p.join()

    answers = [answer_async.get() for answer_async in answers_async if answer_async.ready()]
    res = [pres for pans, pres in answers]
    return res
