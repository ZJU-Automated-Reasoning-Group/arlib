import logging
import random
from typing import List

from pysat.formula import CNF
from pysat.solvers import Solver

logger = logging.getLogger(__name__)


class BoolExistsSolver(object):
    def __init__(self, exists_vars, clauses):
        self.solver_name = "cadical"
        self.solver = Solver(name=self.solver_name, bootstrap_with=clauses)
        self.existential_bools = exists_vars
        # self.universal_bool = forall_vars
        self.clauses = []  # just for debugging?

    def get_random_assignment(self):
        return [v if random.random() < 0.5 else -v for v in self.existential_bools]

    def get_models(self, num=1):
        if len(self.clauses) == 0:
            return [self.get_random_assignment() for _ in range(num)]

        results = []
        if num == 1:
            if self.solver.solve():
                model = self.solver.get_model()
                existential_model = []
                for val in model:  # project?
                    if abs(val) in self.existential_bools:
                        existential_model.append(val)
                logger.debug("model: {}".format(model))
                logger.debug("e-model: {}".format(existential_model))
                results.append(existential_model)
            else:
                logger.debug("e-solver UNSAT")
                logger.debug("clauses: {}".format(self.clauses))
        else:
            for i, model in enumerate(self.solver.enum_models(), 1):
                existential_model = []
                for val in model:  # project?
                    if abs(val) in self.existential_bools:
                        existential_model.append(val)
                results.append(existential_model)
                if i == num:
                    break
        return results

    def add_clause(self, clause: List[int]):
        self.solver.add_clause(clause)
        self.clauses.append(clause)

    def add_clauses(self, clauses: List[List[int]]):
        """Update self.clauses
        E.g., refinement from the ForAllSolver
        """
        for cls in clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)

    def add_cnf(self, cnf: CNF):
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)
