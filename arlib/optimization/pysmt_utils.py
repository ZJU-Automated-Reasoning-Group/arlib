from typing import List
import z3
from pysmt.shortcuts import BVULT, BVUGT
from pysmt.shortcuts import Symbol, And, BV, BVUGE, BVULE, Solver, ForAll, Exists, qelim
from pysmt.typing import INT, REAL, BVType, BOOL
from arlib.utils import get_expr_vars

# BV1, BV8, BV16, BV32, BV64, BV128
# NOTE: both pysmt and z3 have a class "Solver"


def z3_to_pysmt_vars(z3vars: [z3.ExprRef]):
    res = []
    for v in z3vars:
        if z3.is_int(v):
            res.append(Symbol(v.decl().name(), INT))
        elif z3.is_real(v):
            res.append(Symbol(v.decl().name(), REAL))
        elif z3.is_bv(v):
            res.append(Symbol(v.decl().name(), BVType(v.sort().size())))
        elif z3.is_bool(v):
            res.append(Symbol(v.decl().name(), BOOL))
        else:
            raise NotImplementedError
    return res


def z3_to_pysmt(zf: z3.ExprRef, obj: z3.ExprRef):
    # FIXME: we use  the following two lines to hide warnings from PYSMT(?)
    #  However, they seem not to be necessary and z3.z3util.get_vars can be very slow
    #  (Is the warning caused py pySMT?)
    # zvs = z3.z3util.get_vars(zf)  # this can be very slow...
    zvs = get_expr_vars(zf)

    _ = z3_to_pysmt_vars(zvs)

    #
    z3s = Solver(name='z3')
    pysmt_var = Symbol(obj.decl().name(), BVType(obj.sort().size()))
    pysmt_fml = z3s.converter.back(zf)
    return pysmt_var, pysmt_fml
    # return pysmt_vars, pysmt_fml


def quantifier_elimination(qexp):
    return qelim(qexp)
