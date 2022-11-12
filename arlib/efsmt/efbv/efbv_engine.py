"""
This file exports external interfaces of different algorithms for solving
 exists-forall bit-vector problems

 TODO: use pysat.CNF or "raw" numerical clauses?
"""

import logging
from typing import List

import z3

from arlib.efsmt.efbv.efbv_formula_manager import EFBVFormulaTranslator
from arlib.efsmt.efbv.efbv_utils import EFBVResult, EFBVTactic
from arlib.efsmt.efbv.efbv_solver import bv_efsmt_with_uniform_sampling

logger = logging.getLogger(__name__)

g_efbv_tactic = EFBVTactic.Z3_QBF


def solve_with_qbf(fml: z3.ExprRef) -> EFBVResult:
    """Solve Exists X Forall Y Exists Z . P(...), which is translated from an exists-forall bit-vector instance
    NOTE: We do not need to explicitly specify the first Exists
    Z: the aux Boolean vars (e.g., introduced by the bit-blasting and CNF transformer?)
    """
    sol = z3.Solver()
    sol.add(fml)
    res = sol.check()
    if res == z3.sat:
        return EFBVResult.SAT
    elif res == z3.unsat:
        return EFBVResult.UNSAT
    else:
        return EFBVResult.UNKNOWN


def solve_with_simple_cegar(x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None) -> EFBVResult:
    """
    Solve exists-forall bit-vectors
     (The name of the engine is EFBVTactic.SIMPLE_CEGAR)
    """
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

            # the following lines should be done by the forall solver?
            mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
            sub_phi = z3.simplify(z3.substitute(phi, mappings))

            fsolver.push()
            fsolver.add(z3.Not(sub_phi))
            if fsolver.check() == z3.sat:
                fmodel = fsolver.model()
                # the following operations should be sequential?
                # the following line should not be dependent on z3?
                y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                esolver.add(sub_phi)
                fsolver.pop()
            else:
                return EFBVResult.SAT
    return EFBVResult.UNKNOWN


def solve_efsmt_bv(existential_vars: List, universal_vars: List, phi: z3.ExprRef):
    """ Solves exists x. forall y. phi(x, y)
    """
    global g_efbv_tactic
    g_efbv_tactic = EFBVTactic.SEQ_CEGAR

    if g_efbv_tactic == EFBVTactic.Z3_QBF:
        fml_manager = EFBVFormulaTranslator()
        return solve_with_qbf(fml_manager.to_qbf(phi, existential_vars, universal_vars))
    elif g_efbv_tactic == EFBVTactic.SIMPLE_CEGAR:
        return solve_with_simple_cegar(existential_vars, universal_vars, phi)
    elif g_efbv_tactic == EFBVTactic.SEQ_CEGAR:
        return bv_efsmt_with_uniform_sampling(existential_vars, universal_vars, phi)


