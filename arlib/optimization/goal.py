"""Goal in OMT (modified) from pySMT"""

# This file is part of pySMT.
#
#   Copyright 2014 Andrea Micheli and Marco Gario
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

from pysmt.environment import get_env
from pysmt.oracles import get_logic
from pysmt.logics import LIA, LRA, BV
from pysmt.formula import FormulaManager

class OMTFormulaManager(FormulaManager):

    def __int__(self):
        super.__init__()

    def _MinWrap(self, le, *args):
        """Returns the encoding of the minimum expression within args using the specified 'Lower-Equal' operator"""
        exprs = self._polymorph_args_to_tuple(args)
        assert len(exprs) > 0
        if len(exprs) == 1:
            return exprs[0]
        elif len(exprs) == 2:
            a, b = exprs
            return self.Ite(le(a, b), a, b)
        else:
            h = len(exprs) // 2
            return self._MinWrap(le, self._MinWrap(le, exprs[0:h]), self._MinWrap(le, exprs[h:]))

    def _MaxWrap(self, le, *args):
        """Returns the encoding of the maximum expression within args using the specified 'Lower-Equal' operator"""
        exprs = self._polymorph_args_to_tuple(args)
        assert len(exprs) > 0
        if len(exprs) == 1:
            return exprs[0]
        elif len(exprs) == 2:
            a, b = exprs
            return self.Ite(le(a, b), b, a)
        else:
            h = len(exprs) // 2
            return self._MaxWrap(le, self._MaxWrap(le, exprs[0:h]), self._MaxWrap(le, exprs[h:]))

    def MinBV(self, sign, *args):
        """Returns the encoding of the minimum expression within args"""
        le = self.BVULE
        if sign:
            le = self.BVSLE
        return self._MinWrap(le, *args)

    def MaxBV(self, sign, *args):
        """Returns the encoding of the maximum expression within args"""
        le = self.BVULE
        if sign:
            le = self.BVSLE
        return self._MaxWrap(le, *args)

    def Min(self, *args):
        """Returns the encoding of the minimum expression within args"""
        return self._MinWrap(self.LE, *args)

    def Max(self, *args):
        """Returns the encoding of the maximum expression within args"""
        return self._MaxWrap(self.LE, *args)



class Goal:
    """
    This class defines goals for solvers.
    Warning: this class is not instantiable

    Examples:

        example of minimization:
        ```
        with Optimizer(name = "z3") as opt:
            x = Symbol("x", INT)
            min = MinimizationGoal(x)
            formula = GE(y, Int(5))
            opt.add_assertion(formula)
            model, cost = opt.optimize(min)
        ```

        example of maximization:
        ```
        with Optimizer(name = "z3") as opt:
            max = MaximizationGoal(x)
            formula = LE(y, Int(5))
            opt.add_assertion(formula)
            model, cost = opt.optimize(max)
        ```
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
        logic = get_logic(self.formula)
        if logic <= LIA:
            return LIA
        elif logic <= LRA:
            return LRA
        elif logic <= BV:
            return BV
        else:
            return logic

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


class MinMaxGoal(MinimizationGoal):
    """
    Minimize the maximum expression within 'terms'
    This goal is common to all solvers.
    The object can be passed as an argument to the optimize method of any Optimizer
    Warning: some Optimizer may not support this goal
    """

    def __init__(self, terms, sign=False):
        """
        :param terms: List of FNode
        """
        # TODO: a "hack" to create an object from a base class object
        #  Maybe we have better ways..
        omt_fml_manager = object.__new__(OMTFormulaManager)
        omt_fml_manager.__dict__ = get_env().fomula_manager.__dict__.copy()

        if len(terms) > 0:
            if get_env().stc.get_type(terms[0]).is_bv_type():
                formula = omt_fml_manager.MaxBV(sign, terms)
            else:
                formula = omt_fml_manager.Max(terms)
        else:
            formula = omt_fml_manager.Max(terms)

        MinimizationGoal.__init__(self, formula)
        self.terms = terms
        self._bv_signed = sign

    def is_minmax_goal(self):
        return True


class MaxMinGoal(MaximizationGoal):
    """
    Maximize the minimum expression within 'terms'
    This goal is common to all solvers.
    The object can be passed as an argument to the optimize method of any Optimizer
    Warning: some Optimizer may not support this goal
    """

    def __init__(self, terms, sign=False):
        """
        :param terms: List of FNode
        """
        # TODO: a "hack" to create an object from a base class object
        #  Maybe we have better ways..
        omt_fml_manager = object.__new__(OMTFormulaManager)
        omt_fml_manager.__dict__ = get_env().fomula_manager.__dict__.copy()

        if len(terms) > 0:
            if get_env().stc.get_type(terms[0]).is_bv_type():
                formula = omt_fml_manager.MinBV(sign, terms)
            else:
                formula = omt_fml_manager.Min(terms)
        else:
            formula = omt_fml_manager.Min(terms)
        MaximizationGoal.__init__(self, formula)
        self.terms = terms
        self._bv_signed = sign

    def is_maxmin_goal(self):
        return True


class MaxSMTGoal(Goal):
    """
    MaxSMT goal common to all solvers.
    """

    _instance_id = 0

    def __init__(self):
        """Accepts soft clauses and the relative weights"""
        self.id = MaxSMTGoal._instance_id
        MaxSMTGoal._instance_id = MaxSMTGoal._instance_id + 1
        self.soft = []
        self._bv_signed = False

    def add_soft_clause(self, clause, weight):
        """Accepts soft clauses and the relative weights"""
        self.soft.append((clause, weight))

    def is_maxsmt_goal(self):
        return True
