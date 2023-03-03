"""
Solving Exits-Forall Problem (currently focus on bit-vec?)
"""

import logging
from typing import List
import time

import z3

from arlib.efsmt.efbv.blasting_efbv.efbv_to_bool import EFBVFormulaTranslator
from arlib.efsmt.efbv.efbv_utils import EFBVResult, EFBVTactic, EFBVSolver

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


def solve_with_simple_cegar(x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    """
    Solve exists-forall bit-vectors
    """
    # set_param("verbose", 15)
    qf_logic = "QF_BV"  # or QF_UFBV
    esolver = z3.SolverFor(qf_logic)
    fsolver = z3.SolverFor(qf_logic)
    esolver.add(z3.BoolVal(True))
    loops = 0
    while maxloops is None or loops <= maxloops:
        logger.debug("  Round: {}".format(loops))
        loops += 1
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


def solve_with_z3(universal_vars: List, phi: z3.ExprRef):
    sol = z3.SolverFor("UFBV")
    sol.add(z3.ForAll(universal_vars, phi))
    res = sol.check()
    if res == z3.unsat:
        return EFBVResult.UNSAT
    elif res == z3.sat:
        return EFBVResult.SAT
    else:
        return EFBVResult.UNKNOWN


class SequentialEFBVSolver(EFBVSolver):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        qbf, simple_cegar, z3
        """
        self.mode = kwargs.get("mode", "qbf")  # qbf is the default one

    def solve_efsmt_bv(self, existential_vars: List, universal_vars: List, phi: z3.ExprRef):
        if self.mode == "qbf":
            fml_manager = EFBVFormulaTranslator()
            return solve_with_qbf(fml_manager.to_qbf(phi, existential_vars, universal_vars))
        elif self.mode == "z3":
            return solve_with_z3(universal_vars, phi)
        elif self.mode == "simple_cegar":
            return solve_with_simple_cegar(existential_vars, universal_vars, phi)
        else:
            return solve_with_z3(universal_vars, phi)


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    #
    start = time.time()
    solver = SequentialEFBVSolver(mode="z3")
    print(solver.solve_efsmt_bv([x], [y], fmla))
    print(time.time() - start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_efsmt()
