"""
OMT based on Yices2?
"""

from pysmt.solvers.yices import YicesSolver
from arlib.optimization.optimizer import SUAOptimizerMixin, IncrementalOptimizerMixin

class YicesSUAOptimizer(YicesSolver, SUAOptimizerMixin):
    LOGICS = YicesSolver.LOGICS

class YicesIncrementalOptimizer(YicesSolver, IncrementalOptimizerMixin):
    LOGICS = YicesSolver.LOGICS