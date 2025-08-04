"""Reducing OMT to QSMT"""
import z3
from typing import Union, Any
from arlib.optimization.bin_solver import solve_with_bin_smt
from arlib.optimization.pysmt_utils import ForAll, Exists


def arith_opt_with_qsmt(fml: z3.ExprRef, obj: z3.ExprRef, minimize: bool, solver_name: str) -> str:
    """Quantified Satisfaction based OMT

    Args:
        fml: Z3 formula
        obj: Objective variable
        minimize: Whether to minimize (True) or maximize (False)
        solver_name: Name of the solver to use

    Returns:
        String result from the solver
    """
    is_int = True
    if z3.is_real(obj):
        is_int = False

    if is_int:
        obj_misc = z3.Int(str(obj) + "m")
    else:
        obj_misc = z3.Real(str(obj) + "m")
    new_fml = z3.substitute(fml, (obj, obj_misc))
    if minimize:
        qfml = z3.And(fml,
                      z3.ForAll([obj_misc], z3.Implies(new_fml, obj <= obj_misc)))
    else:
        qfml = z3.And(fml,
                      z3.ForAll([obj_misc], z3.Implies(new_fml, obj_misc <= obj)))

    # print(qfml)
    # TODO: why not allowing for using pySMT?...

    if is_int:
        return solve_with_bin_smt("LIA", qfml=qfml,
                                  obj_name=obj.sexpr(),
                                  solver_name=solver_name)
    else:
        return solve_with_bin_smt("LRA",
                                  qfml=qfml,
                                  obj_name=obj.sexpr(),
                                  solver_name=solver_name)


def demo_qsmt() -> None:
    """Demo function for QSMT-based arithmetic optimization."""
    import time
    x, y, z = z3.Reals("x y z")
    fml = z3.And(y >= 0, y < 10)
    print("start solving")
    res = arith_opt_with_qsmt(fml, y, minimize=True, solver_name="z3")
    print(res)
    start = time.time()
    print("solving time: ", time.time() - start)


if __name__ == '__main__':
    demo_qsmt()
