"""Sequential version
 TODO: use pysat.CNF or "raw" numerical clauses?
"""

import logging
from typing import List

import z3
from z3.z3util import get_vars

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager
from pdsmt.efsmt.efbv.exists_solver import ExistsSolver
from pdsmt.efsmt.efbv.forall_solver import ForAllSolver

logger = logging.getLogger(__name__)


def efsmt_bv_seq(existential_vars: List, universal_vars: List, phi: z3.ExprRef, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    FIXME: inconsistent with efsmt
    """
    fml_manager = EFBVFormulaManager()
    fml_manager.initialize(phi, existential_vars, universal_vars)
    esolver = ExistsSolver(fml_manager)
    fsolver = ForAllSolver(fml_manager)
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("Round: ", loops)
        # TODO: need to make the fist and the subsequent iteration different???
        # TODO: in the uniform sampler, I always call the solver once before xx...
        e_models = esolver.get_models(1)

        if len(e_models) == 0:
            return False
        else:
            sub_phis = []
            reverse_sub_phis = []
            for emodel in e_models:
                tau = [emodel.eval(var, True) for var in existential_vars]
                sub_phi = phi
                for i in range(len(existential_vars)):
                    sub_phi = z3.simplify(z3.substitute(sub_phi, (existential_vars[i], tau[i])))
                sub_phis.append(sub_phi)
                reverse_sub_phis.append(z3.Not(sub_phi))

            fsolver.push()  # currently, do nothing
            res_label, fmodels = fsolver.check(reverse_sub_phis)
            # print(fmodels)
            if 0 in res_label:  # there is at least one unsat Not(sub_phi)
                return z3.sat  # fsolver tells sat
            else:
                # refine using all subphi
                # TODO: the fsolver may return sigma, instead of the models
                for fmodel in fmodels:
                    sigma = [fmodel.eval(vy, True) for vy in universal_vars]
                    sub_phi = phi
                    for j in range(len(universal_vars)):
                        sub_phi = z3.simplify(z3.substitute(sub_phi, (universal_vars[j], sigma[j])))
                    # block all CEX?
                    esolver.fmls.append(sub_phi)
                    fsolver.pop()

    return False


def simple_cegar_efsmt_bv(x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    # set_param("verbose", 15)
    qf_logic = "QF_BV"  # or QF_UFBV
    esolver = z3.SolverFor(qf_logic)
    fsolver = z3.SolverFor(qf_logic)
    esolver.add(z3.BoolVal(True))
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


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fml = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    universal_vars = [y]
    existential_vars = [item for item in get_vars(fml) if item not in universal_vars]
    res = simple_cegar_efsmt_bv(existential_vars, universal_vars, fml)
    print(res)


if __name__ == "__main__":
    test_efsmt()
