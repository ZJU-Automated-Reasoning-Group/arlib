"""
Exceptions related to oMT
"""


class PyomtException(Exception):
    """Base class for all custom exceptions of pySMT"""
    pass


class SolverAPINotFound(PyomtException):
    """The Python API of the selected solver cannot be found."""
    pass


class OMTInfinityError(PyomtException):
    """Infinite value in expressions."""
    pass


class OMTInfinitesimalError(PyomtException):
    """Infinite value in expressions."""
    pass


class OMTUnboundedOptimizationError(PyomtException):
    """Infinite optimal value in optimization."""
    pass


class GoalNotSupportedError(PyomtException):
    """Goal not supported by the solver."""

    def __init__(self, current_solver, goal):
        self.current_solver = current_solver
        self.goal = goal

    def solver(self):
        return self.current_solver

    def goal(self):
        return self.goal
