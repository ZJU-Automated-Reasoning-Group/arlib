"""
Interface for the OMT (Optimization Modulo Theory)
Modified from PySMT
"""

class Optimizer:

    def optimize(self, goal, **kwargs):
        raise NotImplementedError

    def pareto_optimize(self, goals):
        raise NotImplementedError

    def lexicographic_optimize(self, goals):
        raise NotImplementedError

    def boxed_optimize(self, goals):
        raise NotImplementedError


