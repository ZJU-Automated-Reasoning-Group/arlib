# coding: utf-8
from __future__ import print_function
from typing import List
from pysat.solvers import Solver, SolverNames
from pysat.formula import CNF
import random

"""
Wrappers for PySAT
"""

sat_solvers = ['cadical',
               'gluecard30',
               'gluecard41',
               'glucose30',
               'glucose41',
               'lingeling',
               'maplechrono',
               'maplecm',
               'maplesat',
               'minicard',
               'mergesat3',
               'minisat22',
               'minisat-gh']


class PySATSolver(object):
    def __init__(self, solver="cadical"):
        self._solver = Solver(name=solver)

    def check_sat(self):
        return self._solver.solve()

    def add_clause(self, clause: List[int]):
        self._solver.add_clause(clause)

    def add_clauses(self, clauses: List[List]):
        for cls in clauses:
            self._solver.add_clause(cls)

    def add_cnf(self, cnf: CNF):
        self._solver.append_formula(cnf.clauses, no_return=False)
        # for cls in cnf.clauses:
        #    self._solver.add_clause(cls)

    def enumerate_models(self, to_enum: int):
        results = []
        for i, model in enumerate(self._solver.enum_models(), 1):
            results.append(model)
            # print('v {0} 0'.format(' '.join(['{0}{1}'.format('+' if v > 0 else '', v) for v in model])))
            print("i-th model: ", i)
            if i == to_enum:
                break
        return results

    def get_model(self):
        return self._solver.get_model()



def test_pysat():
    cnf = CNF(from_clauses=[[1, 2], [-1, 2], [2]])
    # solver_name = random.choice(sat_solvers)
    s1 = PySATSolver()
    s1.add_cnf(cnf)
    # s1.add_clause([-1, 2])
    # s1.add_clause([-2, 3])
    # s1.add_clause([-3, 4])
    print(s1.check_sat())
    s1.enumerate_models(10)


if __name__ == "__main__":
    test_pysat()
