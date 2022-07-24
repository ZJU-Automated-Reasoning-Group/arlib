"""Sequential version
 TODO: use pysat.CNF or "raw" numerical clauses?
"""

import logging
from typing import List

import z3
from z3.z3util import get_vars

from pdsmt.efsmt.efbv.efbv_formula_manager import EFBVFormulaManager
from pdsmt.efsmt.efbv.efbv_exists_solver import ExistsSolver
from pdsmt.efsmt.efbv.efbv_forall_solver import ForAllSolver
from pdsmt.efsmt.efbv.efbv_utils import EFBVResult


logger = logging.getLogger(__name__)


def solve_qbf(universal_vars: List, fml: z3.ExprRef):
    sol = z3.Solver()
    sol.add(z3.ForAll(universal_vars, fml))
    print(sol.to_smt2())
    res = sol.check()
    if res == z3.sat:
        return EFBVResult.SAT
    elif res == z3.unsat:
        return EFBVResult.UNSAT
    else:
        return EFBVResult.UNKNOWN


def efsmt_bv_seq(existential_vars: List, universal_vars: List, phi: z3.ExprRef, maxloops=None):
    """ Solves exists x. forall y. phi(x, y)
    """
    fml_manager = EFBVFormulaManager()
    fml_manager.initialize(phi, existential_vars, universal_vars)

    u_vars, z3fml = fml_manager.to_z3_clauses(prefix="q")
    # print(u_vars)
    # print(z3fml)
    return solve_qbf(u_vars, z3fml)

    """
    esolver = ExistsSolver(fml_manager)
    fsolver = ForAllSolver(fml_manager)
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("Round: ", loops)
        e_models = esolver.get_models(1)
        if len(e_models) == 0:
            return EFBVResult.UNSAT
        print(e_models)
        blocking_clauses = fsolver.check_models(e_models)
        if len(blocking_clauses) == 0:
            return EFBVResult.SAT
        for cls in blocking_clauses:
            esolver.add_clause(cls)
    return EFBVResult.UNKNOWN
    """


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
            return EFBVResult.UNSAT
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
                return EFBVResult.SAT
    return EFBVResult.UNKNOWN


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 8)
    fml = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    universal_vars = [y]
    existential_vars = [item for item in get_vars(fml) if item not in universal_vars]
    res = simple_cegar_efsmt_bv(existential_vars, universal_vars, fml)
    print(res)


if __name__ == "__main__":
    test_efsmt()
