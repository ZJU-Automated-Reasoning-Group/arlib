"""Sequential version"""

import logging
from typing import List

import z3
from z3.z3util import get_vars

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager
from pdsmt.efsmt.efbv.exists_solver import ExistsSolver
from pdsmt.efsmt.efbv.forall_solver import ForAllSolver

logger = logging.getLogger(__name__)


def efbv_seq(x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef):
    fml_manger = EFBVFormulaManager()
    fml_manger.initialize(phi, x, y)
    esolver = ExistsSolver(fml_manger)
    fsolver = ForAllSolver(fml_manger)

    while True:
        # print("round: ", loops)
        eres = esolver.check_sat()
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
