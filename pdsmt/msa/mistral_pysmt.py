#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# mistral.py based on pysmt
#
#  Created on: Dec 18, 2015
#     Author: Alessandro Previti, Alexey S. Ignatiev
#     E-mail: alessandro.previti@ucdconnect.ie, aignatiev@ciencias.ulisboa.pt
#

#
# ==============================================================================
from pysmt.smtlib.parser import SmtLibParser
from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Bool, get_model, Not, Solver, qelim, ForAll


# ==============================================================================
def get_qmodel(x_univl, formula, maxiters=None, solver_name=None, verbose=False):
    """
        A simple 2QBF CEGAR implementation for SMT.
    """

    x_univl = set(x_univl)
    x_exist = formula.get_free_variables() - x_univl

    with Solver(name=solver_name) as asolver:
        asolver.add_assertion(Bool(True))
        iters = 0

        while maxiters is None or iters <= maxiters:
            iters += 1

            amodel = asolver.solve()
            if not amodel:
                return None
            else:
                cand = {v: asolver.get_value(v) for v in x_exist}
                subform = formula.substitute(cand).simplify()
                if verbose:
                    print('c qsolve cand{0}: {1}'.format(iters, cand))

                cmodel = get_model(Not(subform), solver_name=solver_name)
                if cmodel is None:
                    return cand
                else:
                    coex = {v: cmodel[v] for v in x_univl}
                    subform = formula.substitute(coex).simplify()
                    if verbose:
                        print('c qsolve coex{0}: {1}'.format(iters, coex))

                    asolver.add_assertion(subform)

        raise SolverReturnedUnknownResultError


#
# ==============================================================================
class Mistral:
    """
        Mistral solver class.
    """

    def __init__(self, simplify, solver, qsolve, verbose, fname):
        """
            Constructor.
        """

        self.script = SmtLibParser().get_script_fname(fname)
        self.formula = self.script.get_last_formula()
        if simplify:
            self.formula = self.formula.simplify()
        self.fvars = self.formula.get_free_variables()

        self.cost = 0
        self.sname = solver
        self.verb = verbose
        self.qsolve = qsolve

        if self.verb > 2:
            print('c formula: \'{0}\''.format(self.formula))

        if self.verb > 1:
            print('c vars ({0}):'.format(len(self.fvars)), list(self.fvars))

    def solve(self):
        """
            This method implements find_msa() procedure from Fig. 2
            of the dillig-cav12 paper.
        """

        # testing if formula is satisfiable
        if not get_model(self.formula, solver_name=self.sname):
            return None

        mus = self.compute_mus(frozenset([]), self.fvars, 0)

        model = self.get_model_forall(mus)
        return ['{0}={1}'.format(v, model[v]) for v in self.fvars - mus]

    def compute_mus(self, X, fvars, lb):
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

        Y = self.compute_mus(X, fvars - x, lb)
        if len(Y) > lb:
            best = Y

        return best

    def get_model_forall(self, x_univl):
        """
            Calls either pysmt.shortcuts.get_model() or get_qmodel().
        """

        if self.qsolve == 'std':
            return get_model(ForAll(x_univl, self.formula),
                             solver_name=self.sname)
        elif self.qsolve == 'z3qe':
            formula = qelim(ForAll(x_univl, self.formula))
            return get_model(formula, solver_name=self.sname)
        else:
            return get_qmodel(x_univl, self.formula, solver_name=self.sname,
                              verbose=True if self.verb > 2 else False)
