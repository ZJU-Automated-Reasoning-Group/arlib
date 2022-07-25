from typing import List
import logging

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager

logger = logging.getLogger(__name__)


class ForAllSolver(object):
    def __init__(self, manager: EFBVFormulaManager):
        self.fml_manager = manager

    def check_models(self, models: List[List]):
        raise NotImplementedError

