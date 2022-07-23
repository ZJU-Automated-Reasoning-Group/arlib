from typing import List
import copy

from pysat.solvers import Solver
from pysat.formula import CNF

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager


sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41', 'lingeling',
               'maplechrono', 'maplecm', 'maplesat', 'minicard', 'mergesat3', 'minisat22', 'minisat-gh']


class ForAllSolver(object):
    def __init__(self, manager: EFBVFormulaManager):
        self.solver_name = "cadical"
        # self._solver = Solver(name=self.solver_name)
        self.fml_manager = manager
        self._clauses = copy.deepcopy(manager.bool_clauses)  # should we do this?

    def check_models(self):
        return self._solver.solve()

    def refine(self):
        raise NotImplementedError