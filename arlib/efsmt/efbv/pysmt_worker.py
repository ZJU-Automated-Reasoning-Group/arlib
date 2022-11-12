# coding: utf-8
import z3
from pysmt.oracles import get_logic
from pysmt.shortcuts import EqualsOrIff
from pysmt.shortcuts import Symbol, And
from pysmt.typing import INT, REAL, BOOL, BVType, BV1, BV8, BV16, BV32, BV64
from pysmt.shortcuts import Not, Solver


# NOTE: both pysmt and z3 have a class "Solver"
# logger = logging.getLogger(__name__)


def convert(zf: z3.ExprRef):
    """
    FIXME: if we do not call "pysmt_vars = ...", z3 will report naming warning..
    """
    zvs = z3.z3util.get_vars(zf)
    pysmt_vars = [Symbol(v.decl().name(), INT if v.is_int() else REAL) for v in zvs]
    z3s = Solver(name='z3')
    pysmt_fml = z3s.converter.back(zf)
    return pysmt_vars, pysmt_fml


def to_pysmt_vars(z3vars: [z3.ExprRef]):
    return [Symbol(v.decl().name(),
                   INT if v.is_int() else REAL) for v in z3vars]


def check_sat_with_pysmt(fml, vars):
    _, pysmt_fml = convert(fml)
    target_logic = get_logic(pysmt_fml)

    pysmt_vars = to_pysmt_vars(vars)
    # print("Target Logic: %s" % target_logic)

    with Solver(logic=target_logic) as solver:
        solver.add_assertion(pysmt_fml)
        res = solver.solve()
        if res:
            model = [EqualsOrIff(k, solver.get_value(k)) for k in pysmt_vars]
            # TODO: map back to z3 world?


"""
def pysmt_binary_itp(fml_a: z3.ExprRef, fml_b: z3.ExprRef) -> z3.ExprRef:
    _, pysmt_fml_a = to_pysmt_fml(fml_a)
    _, pysmt_fml_b = to_pysmt_fml(fml_b)

    itp = Interpolator(name="msat")
    res = itp.binary_interpolant(pysmt_fml_a, pysmt_fml_b)
    return Solver(name='z3').converter.convert(res)
"""