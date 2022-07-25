import random
from typing import List
import logging

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager

logger = logging.getLogger(__name__)


class EFBVExistsSolver(object):
    def __init__(self, manager: EFBVFormulaManager):
        self.solver_name = "cadical"
        self.fml_manager = manager
        # self.solver = Solver(name=self.solver_name, bootstrap_with=manager.bool_clauses)

    def get_random_assignment(self):
        return [v if random.random() < 0.5 else -v for v in self.fml_manager.existential_bools]

    def get_models(self, num=1):
        """Generate more or more assignments (for the existential values)
         TODO:
          -  Unigen and other third-party (uniform) samplers
          -  Allow for passing a set of support variables
        """
        raise NotImplementedError
