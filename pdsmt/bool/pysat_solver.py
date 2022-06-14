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
        self.solver = Solver(name=solver)
        self.clauses = []

    def check_sat(self):
        return self.solver.solve()

    def add_clause(self, clause: List[int]):
        self.solver.add_clause(clause)
        self.clauses.append(clause)

    def add_clauses(self, clauses: List[List]):
        for cls in clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)

    def add_cnf(self, cnf: CNF):
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)

    def sample_models(self, to_enum: int):
        results = []
        for i, model in enumerate(self.solver.enum_models(), 1):
            results.append(model)
            if i == to_enum:
                break

        return results

    def reduce_models(self, models: List[List]):
        """
        http://fmv.jku.at/papers/NiemetzPreinerBiere-FMCAD14.pdf
        Consider a Boolean formula P. The model of P (given by a SAT solver) is not necessarily minimal.
        In other words, the SAT solver may assign truth assignments to literals irrelevant to truth of P.

        Suppose we have a model M of P. To extract a smaller assignment, one trick is to encode the
        negation of P in a separate dual SAT solver.

        We can pass M as an assumption to the dual SAT solver. (check-sat-assuming M).
        All assumptions inconsistent with -P (called the failed assumptions),
        are input assignments sufficient to falsify -P, hence sufficient to satisfy P.

        Related work
          - https://arxiv.org/pdf/2110.12924.pdf
        """
        pos = CNF(from_clauses=self.clauses)
        neg = pos.negate()
        # print(neg.clauses) print(neg.auxvars)
        reduced_models = []
        aux_sol = Solver(name="cadical", bootstrap_with=neg)
        for m in models:
            assert not aux_sol.solve(m)
            reduced_models.append(aux_sol.get_core())
        return reduced_models

    def get_model(self):
        return self.solver.get_model()


def test_pysat():
    cnf = CNF(from_clauses=[[1, 3], [-1, 2, -4], [2, 4]])
    # solver_name = random.choice(sat_solvers)
    s1 = PySATSolver()
    s1.add_cnf(cnf)
    models = s1.sample_models(10)
    print(models)
    # many reduced models are duplicate
    print(s1.reduce_models(models))


if __name__ == "__main__":
    test_pysat()
