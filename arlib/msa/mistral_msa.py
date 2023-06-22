"""
This module provides an implementation of the Minimal Satisfying Assignment (MSA) algorithm,
adapted from the algorithm by Alessandro Previti and Alexey S. Ignatiev. It contains the MSASolver
class which is used to find the minimal satisfying assignment for a given formula.
"""
from typing import FrozenSet

import z3
from z3.z3util import get_vars

class MSASolver:
    """
    Mistral solver class.
    """

    def __init__(self, verbose=1):
        """
        Constructor.
        """
        self.formula = None
        # self.formula = simplify(self.formula)
        self.fvars = None
        self.verb = verbose

    def init_from_file(self, filename: str) -> None:
        self.formula = z3.And(z3.parse_smt2_file(filename))
        # self.formula = simplify(self.formula)
        self.fvars = frozenset(get_vars(self.formula))

        if self.verb > 2:
            print('c formula: \'{0}\''.format(self.formula))

    def init_from_formula(self, formula: z3.BoolRef) -> None:
        self.formula = formula
        # self.formula = simplify(self.formula)
        self.fvars = frozenset(get_vars(self.formula))

        if self.verb > 2:
            print('c formula: \'{0}\''.format(self.formula))

    def validate_small_model(self, model: z3.ModelRef) -> bool:
        """Check whether a small model is a 'sufficient condition'"""
        decls = model.decls()
        model_cnts = []
        for var in get_vars(self.formula):
            if var.decl() in decls:
                model_cnts.append(var == model[var])
        # print(model_cnts)
        # check entailment
        s = z3.Solver()
        s.add(z3.Not(z3.Implies(z3.And(model_cnts), self.formula)))
        if s.check() == z3.sat:
            return False
        return True

    def find_small_model(self):
        """
        This method implements find_msa() procedure from Fig. 2
        of the dillig-cav12 paper.
        """
        # testing if formula is satisfiable
        s = z3.Solver()
        s.add(self.formula)
        if s.check() == z3.unsat:
            return False

        mus = self.compute_mus(frozenset([]), self.fvars, 0)

        model = self.get_model_forall(mus)
        return model
        # return ['{0}={1}'.format(v, model[v]) for v in frozenset(self.fvars) - mus]

    def compute_mus(self, X: FrozenSet, fvars: FrozenSet, lb: int):
        """
        Algorithm implements find_mus() procedure from Fig. 1
        of the dillig-cav12 paper.
        """

        if not fvars or len(fvars) <= lb:
            return frozenset()

        best = set()
        x = frozenset([next(iter(fvars))])  # should choose x in a more clever way

        if self.verb > 1:
            print('c state:', 'X = {0} + {1},'.format(list(X), list(x)), 'lb =', lb)

        if self.get_model_forall(X.union(x)):
            Y = self.compute_mus(X.union(x), fvars - x, lb - 1)

            cost_curr = len(Y) + 1
            if cost_curr > lb:
                best = Y.union(x)
                lb = cost_curr

        Y = self.compute_mus(X, frozenset(fvars) - x, lb)
        if len(Y) > lb:
            best = Y

        return best

    def get_model_forall(self, x_univl):
        s = z3.Solver()
        if len(x_univl) >= 1:
            qfml = z3.ForAll(list(x_univl), self.formula)
        else:
            qfml = self.formula  # TODO: is this OK?
        s.add(qfml)
        if s.check() == z3.sat:
            return s.model()
        return False


if __name__ == "__main__":
    a, b, c, d = z3.Ints('a b c d')
    fml = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))
    ms = MSASolver()
    ms.init_from_formula(fml)
    print(ms.find_small_model())  # a = 3, b = 3
    # qfml = ForAll([c, d], fml)
    # s = Solver()
    # s.add(qfml)
    # s.check()
    # print(s.model())  # [a = 3, b = 3]
