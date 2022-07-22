"""
Common facilities for Boolean-level reasoning
"""
from multiprocessing import Pool
from typing import List, Tuple

from pysat.solvers import Solver

sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41', 'lingeling',
               'maplechrono', 'maplecm', 'maplesat', 'minicard', 'mergesat3', 'minisat22', 'minisat-gh']


def check_sat_assuming(clauses: List[List[int]], assumptions: List[int]) -> Tuple:
    """Used by parallel solving"""
    solver = Solver(name="cadical", bootstrap_with=clauses)
    ans = solver.solve(assumptions=assumptions)
    if ans:
        return ans, solver.get_model()
    return ans, solver.get_core()


def parallel_solve_assumptions(clauses: List[List[int]], assumptions_lists: List[List[int]]):
    """Solve clauses under a set of assumptions (deal with each one in parallel)
    TODO: - Should we enforce that clauses are satisfiable?
          - Should control size of the Pool
          - Add timeout (if timeout, use the original model?)
    """
    answers_async = [None for _ in assumptions_lists]
    with Pool(len(assumptions_lists)) as p:
        def terminate_others(val):
            if val:
                p.terminate()

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
