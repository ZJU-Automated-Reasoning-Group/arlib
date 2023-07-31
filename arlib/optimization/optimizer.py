"""Interfaces for Optimization Modulo Theory (OMT) Solving"""


class Optimizer:
    """
    Interface for the optimization
    """

    def optimize(self, goal, **kwargs):
        """Returns a pair `(model, cost)` where `model` is an object
        that obtained according to `goal` while satisfying all
        the formulae asserted in the optimizer, while `cost` is the
        objective function value for the model.

        `goal` must have a term with integer, real or
        bit-vector type whose value has to be minimized
        This function can raise a PysmtUnboundedOptimizationError if
        the solver detects that the optimum value is either positive
        or negative infinite or if there is no optimum value because
        one can move arbitrarily close to the optimum without reching
        it (e.g. "x > 5" has no minimum for x, only an infimum)

        """
        raise NotImplementedError


    def pareto_optimize(self, goals):
        """This function is a generator returning *all* the pareto-optimal
        solutions for the problem of minimizing the `cost_functions`
        keeping the formulae asserted in this optimizer satisfied.

        The solutions are returned as pairs `(model, costs)` where
        model is the pareto-optimal assignment and costs is the list
        of costs, one for each optimization function in
        `cost_functions`.

        `cost_functions` must be a list of terms with integer, real or
        bit-vector types whose values have to be minimized

        This function can raise a PysmtUnboundedOptimizationError if
        the solver detects that the optimum value is either positive
        or negative infinite or if there is no optimum value because
        one can move arbitrarily close to the optimum without reching
        it (e.g. "x > 5" has no minimum for x, only an infimum)
        """
        raise NotImplementedError


    def lexicographic_optimize(self, goals):
        """
        This function returns a pair of (model, values) where 'values' is a list containing the optimal values
        (as pysmt constants) for each goal in 'goals'.
        If there is no solution the function returns a pair (None,None)

        The parameter 'goals' must be a list of 'Goals'(see file goal.py).

        The lexicographic method consists of solving a sequence of single-objective optimization problems. The order of
        problems are important because the result of previous goals become a costraint for subsequent goals

        For some implemented examples see file pysmt/test/test_optimization.py
        """
        raise NotImplementedError


    def boxed_optimize(self, goals):
        """
        This function returns dictionary where the keys are the goals of optimization. Each goal is mapped to a pair
        (model, value) of the current goal (key).
        If there is no solution the function returns None

        The parameter 'goals' must be a list of 'Goals'(see file goal.py).

        The boxed method consists of solving a list of single-objective optimization problems independently of
        each other. The order of problems isn't important because the result of previous goals don't change
        any constraint of other goals

        For some implemented examples see file pysmt/test/test_optimization.py
        """
        raise NotImplementedError


    def can_diverge_for_unbounded_cases(self):
        """This function returns True if the algorithm implemented in this
        optimizer can diverge (i.e. not terminate) if the objective is
        unbounded (infinite or infinitesimal).
        """
        raise NotImplementedError


    def _get_symbol_type(self, objective_formula):
        raise NotImplementedError

    def _get_or(self, objective_formula):
        raise NotImplementedError


    def _get_le(self, objective_formula):
        raise NotImplementedError


    def _get_lt(self, objective_formula):
        raise NotImplementedError


class OptComparationFunctions:

    def _comparation_functions(self, goal):
        """Internal utility function to get the proper cast, LT and LE
        function for the given objective formula
        """
        raise NotImplementedError
