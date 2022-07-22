from typing import List
import random

from pysat.solvers import Solver
from pysat.formula import CNF

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager

sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41', 'lingeling',
               'maplechrono', 'maplecm', 'maplesat', 'minicard', 'mergesat3', 'minisat22', 'minisat-gh']


class ExistsSolver(object):
    def __init__(self, manager: EFBVFormulaManager):
        self.solver_name = "cadical"
        self.fml_manager = manager
        self._solver = Solver(name=self.solver_name)
        self._clauses = [] # do not add fml_manager.bool_clauses here

    def get_random_assignment(self):
        """Randomly assign values to the existential variables
        result = []
        for v in self.fml_manager.existential_bools:
            if random.random() < 0.5:
                result.append(v)
            else:
                result.append(-v)
        return result
        """
        return [v if random.random() < 0.5 else -v for v in self.fml_manager.existential_bools]

    def check_sat(self):
        """Generate more or more assignments (for the existential values)
        NOTE: in the first round, we use randomly assign values to the existential varialbes?
        """
        if len(self._clauses) == 0:  # the first round
            return self.get_random_assignment()
        else:
            return self._solver.solve()

    def add_clause(self, clause: List[int]):
        self._solver.add_clause(clause)
        self._clauses.append(clause)

    def add_clauses(self, clauses: List[List[int]]):
        """Update self._clauses
        E.g., refinement from the ForAllSolver
        """
        for cls in clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def add_cnf(self, cnf: CNF):
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def get_model(self):
        return self._solver.get_model()


def test_prop():
    cnf = CNF(from_clauses=[[1, 3], [-1, 2, -4], [2, 4]])
    # solver_name = random.choice(sat_solvers)
    sol = ExistsSolver()
    sol.add_cnf(cnf)
    print(sol.check_sat())