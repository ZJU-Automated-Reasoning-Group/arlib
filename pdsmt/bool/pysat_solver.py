# coding: utf-8
from __future__ import print_function
from typing import List
from pysat.solvers import Solver  # standard way to import the library
from pysat.formula import CNF
import random

"""
Wrappers for PySAT
"""


class PySATSolver:
    def __init__(self, name="cadical"):
        """
        Build a smtlib solver instance.
        This is implemented using an external solver (via a subprocess).
        """
        self._solver = Solver(name=name)

    def check_sat(self):
        return self._solver.solve()

    def add_clause(self, clause: List[int]):
        self._solver.add_clause(clause)

    def add_clauses(self, clauses: List[List]):
        for clause in clauses:
            self._solver.add_clause(clause)

    def add_clauses_from_string(self, cnfstr: str):
        cnf = CNF(from_string=cnfstr)
        print(cnf.clauses)


def test():
    s1 = PySATSolver()
    s1.add_clause([-1, 2])
    s1.add_clause([-2, 3])
    s1.add_clause([-3, 4])
    print(s1.check_sat())


# test()

solvers = ['cadical',
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


def test_cnf():
    cnf = CNF(from_clauses=[[1, 2], [1, 2], [2]])
    solver_name = random.choice(solvers)

    with Solver(name=solver_name, bootstrap_with=cnf) as solver:
        print(solver.solve(), 'outcome by {0}'.format(solver_name))
        for i in solver.enum_models():
            print(i)


test_cnf()
