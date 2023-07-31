"""Goal in OMT"""

class Goal:
    """
    This class defines goals for solvers.
    Warning: this class is not instantiable
    """

    def is_maximization_goal(self):
        return False

    def is_minimization_goal(self):
        return False

    def is_minmax_goal(self):
        return False

    def is_maxmin_goal(self):
        return False

    def is_maxsmt_goal(self):
        return False

    def get_logic(self):
        raise NotImplementedError

    @property
    def signed(self):
        return self._bv_signed

    @signed.setter
    def signed(self, value):
        self._bv_signed = value


class MaximizationGoal(Goal):
    """
    Maximization goal common to all solvers.
    The object can be passed as an argument to the optimize method of any Optimizer
    Warning: some Optimizer may not support this goal
    """

    def __init__(self, formula, signed=False):
        """
        :param formula: The target formula
        :type  formula: FNode
        """
        self.formula = formula
        self._bv_signed = signed

    def opt(self):
        return MaximizationGoal

    def term(self):
        return self.formula

    def is_maximization_goal(self):
        return True


class MinimizationGoal(Goal):
    """
    Minimization goal common to all solvers.
    The object can be passed as an argument to the optimize method of any Optimizer
    Warning: some Optimizer may not support this goal
    """

    def __init__(self, formula, sign=False):
        """
        :param formula: The target formula
        :type  formula: FNode
        """
        self.formula = formula
        self._bv_signed = sign

    def opt(self):
        return MinimizationGoal

    def term(self):
        return self.formula

    def is_minimization_goal(self):
        return True
