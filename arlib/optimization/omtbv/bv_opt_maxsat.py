"""
Reduce OMT(BV) to Weighted MaxSAT

1. OBV-BS and its variants
2. Existing weighted MaxSAT...
"""
import logging
from typing import List, Union, Any

import z3

from arlib.optimization.omtbv.bit_blast_omt_solver import BitBlastOMTBVSolver

logger = logging.getLogger(__name__)


def bv_opt_with_maxsat(z3_fml: z3.ExprRef, z3_obj: z3.ExprRef,
                       minimize: bool, solver_name: str) -> Union[int, float]:
    """Reduce OMT(BV) to Weighted MaxSAT

    Args:
        z3_fml: Z3 formula
        z3_obj: Objective variable
        minimize: Whether to minimize (True) or maximize (False)
        solver_name: Name of the MaxSAT solver to use

    Returns:
        Optimal value found by the solver
    """
    omt = BitBlastOMTBVSolver()
    omt.from_smt_formula(z3_fml)
    omt.set_engine(solver_name)
    sz = z3_obj.size()
    max_bv = (1 << sz) - 1
    if minimize:
        # FIXME: it seems that we convert all the objectives to "maximize xx".
        #  So, maybe we do not need this new API? But how can we know whether the original
        #  objective is "minimize" or "maximize"?
        # TODO: add the API minimize_with_maxsat???
        # return omt.minimize_with_maxsat(z3_obj, is_signed=False)
        # is the following right?
        tmp = omt.maximize_with_maxsat(-z3_obj, is_signed=False)
        # print(tmp)
        return max_bv + 1 - tmp  # why?
    else:
        return omt.maximize_with_maxsat(z3_obj, is_signed=False)


def demo_maxsat() -> None:
    """Demo function for MaxSAT-based bit-vector optimization."""
    import time
    x, y, z = z3.BitVecs("x y z", 4)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    print("start solving")
    res = bv_opt_with_maxsat(fml, y, minimize=True, solver_name="FM")
    print(res)
    start = time.time()
    print("solving time: ", time.time() - start)


if __name__ == '__main__':
    demo_maxsat()
