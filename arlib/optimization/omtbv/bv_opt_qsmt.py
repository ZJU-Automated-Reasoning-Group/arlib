"""
Reduce OMT(BV) to QSMT and call SMT solvers
that support quantified bit-vector formulas
- Z3
- CVC5
- Q3B
- ...?
"""

import z3
from typing import Union, Any
from arlib.optimization.bin_solver import solve_with_bin_smt
from arlib.optimization.pysmt_utils import ForAll, Exists
from arlib.utils.z3_expr_utils import get_expr_vars


def bv_opt_with_pysmt() -> None:
    """PySMT-based bit-vector optimization (not implemented)."""
    raise NotImplementedError


def bv_opt_with_qsmt(fml: z3.ExprRef, obj: z3.ExprRef, minimize: bool, solver_name: str) -> str:
    """Quantified Satisfaction based OMT for bit-vectors

    Args:
        fml: Z3 formula
        obj: Objective variable
        minimize: Whether to minimize (True) or maximize (False)
        solver_name: Name of the solver to use

    Returns:
        String result from the solver

    # FIXME: it seems that we convert all the objectives to "maximize xx".
       So, maybe we do not need this new API? But how can we know whether the original
       objective is "minimize" or "maximize"?
    """

    objname = obj
    all_vars = get_expr_vars(fml)
    if obj not in all_vars:
        # NOTICE: we create a new variable to represent obj (a term, e.g., x + y)
        objname = z3.BitVec(str(obj), obj.sort().size())
        fml = z3.And(fml, objname == obj)

    obj_misc = z3.BitVec("m_" + str(objname), obj.size())
    new_fml = z3.substitute(fml, (obj, obj_misc))
    # TODO: bvule or < (distinguish between unsigned and signed...)

    if minimize:
        qfml = z3.And(fml,
                      z3.ForAll([obj_misc], z3.Implies(new_fml, z3.ULE(obj, obj_misc))))
    else:
        qfml = z3.And(fml,
                      z3.ForAll([obj_misc], z3.Implies(new_fml, z3.ULE(obj_misc, obj))))
    print(qfml)
    if z3.is_bv(obj):
        return solve_with_bin_smt("BV", qfml=qfml, obj_name=obj.sexpr(), solver_name=solver_name)
    else:
        return solve_with_bin_smt("ALL", qfml=qfml, obj_name=obj.sexpr(), solver_name=solver_name)


def demo_qsmt() -> None:
    """Demo function for QSMT-based bit-vector optimization."""
    import time
    x, y, z = z3.BitVecs("x y z", 16)
    fml = z3.And(z3.UGT(y, 0), z3.ULT(y, 10))
    print("start solving")
    res = bv_opt_with_qsmt(fml, y, minimize=True, solver_name="z3")
    print(res)
    start = time.time()
    print("solving time: ", time.time() - start)


if __name__ == '__main__':
    demo_qsmt()
