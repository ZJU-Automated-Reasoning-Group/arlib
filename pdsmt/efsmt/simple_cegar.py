from typing import List

import z3
from z3.z3util import get_vars


def cegar_efsmt(y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    """
    x = [item for item in get_vars(phi) if item not in y]
    # set_param("verbose", 15)
    # set_param("smt.arith.solver", 3)
    esolver = z3.SolverFor("QF_BV")
    esolver.add(z3.BoolVal(True))

    fsolver = z3.SolverFor("QF_BV")

    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("round: ", loops)
        eres = esolver.check()
        if eres == z3.unsat:
            return z3.unsat
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
                fsolver.pop()
            else:
                return z3.sat
    return z3.unknown
