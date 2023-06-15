"""
Exceptions related to oMT
"""

from pysmt.exceptions import PysmtException


class PysmtInfinityError(PysmtException):
    """Infinite value in expressions."""
    pass

class PysmtInfinitesimalError(PysmtException):
    """Infinite value in expressions."""
    pass

class PysmtUnboundedOptimizationError(PysmtException):
    """Infinite optimal value in optimization."""
    pass


class GoalNotSupportedError(PysmtException):
    """Goal not supported by the solver."""
    def __init__(self, current_solver, goal):
        self.current_solver = current_solver
        self.goal = goal

    def solver(self):
        return self.current_solver

    def goal(self):
        return self.goal

