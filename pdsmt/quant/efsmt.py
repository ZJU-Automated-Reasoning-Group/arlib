# coding: utf-8
import logging
import z3
from z3.z3util import get_vars

logger = logging.getLogger(__name__)

"""
A simple CEAGR-style approach for solving exists x. forall y. phi(x, y)
It can also be understood as a "two-player game"
x is the set of template variables (introduced by the template)
y is the set of "program variables" (used in the original VC)
"""


def efsmt_solve_aux(y, phi, maxloops=None):
    x = [item for item in get_vars(phi) if item not in y]
    esolver = z3.SolverFor("QF_LRA")
    fsolver = z3.SolverFor("QF_LRA")
    esolver.add(z3.BoolVal(True))
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        if esolver.check() == z3.unsat:
            return z3.unsat, False
        else:
            emodel = esolver.model()
            mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
            sub_phi = z3.simplify(z3.substitute(phi, mappings))
            fsolver.push()
            fsolver.add(z3.Not(sub_phi))
            if fsolver.check() == z3.sat:
                fmodel = fsolver.model()
                y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                esolver.add(sub_phi)
                # esolver.add(z3.Tactic("solve-eqs")(sub_phi).as_expr())
                fsolver.pop()
            else:
                return z3.sat, emodel

    return z3.unknown, False
