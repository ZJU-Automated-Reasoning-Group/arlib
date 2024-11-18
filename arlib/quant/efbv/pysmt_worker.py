# coding: utf-8
"""
Map pysmt model to z3 model?
"""
import z3
# from pysmt.oracles import get_logic
from pysmt.shortcuts import EqualsOrIff, get_model
from pysmt.shortcuts import Symbol, And
from pysmt.typing import INT, REAL, BOOL, BVType, BV1, BV8, BV16, BV32, BV64
from pysmt.shortcuts import Not, Solver
from pysmt.logics import QF_BV, QF_LRA, QF_LIA


# NOTE: both pysmt and z3 have a class "Solver"
# logger = logging.getLogger(__name__)


def convert(zf: z3.ExprRef):
    """
    FIXME: if we do not call "pysmt_vars = ...", z3 will report naming warning..
    """
    zvs = z3.z3util.get_vars(zf)
    pysmt_vars = to_pysmt_vars(zvs)
    z3s = Solver(name='z3')
    pysmt_fml = z3s.converter.back(zf)
    return pysmt_vars, pysmt_fml


def to_pysmt_vars(z3vars: [z3.ExprRef]):
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



def check_sat_with_pysmt(fml):
    _, pysmt_fml = convert(fml)
    # pysmt_vars = to_pysmt_vars(vars)
    # target_logic = get_logic(pysmt_fml)
    # print("Target Logic: %s" % target_logic)
    mod = get_model(pysmt_fml, solver_name="z3", logic=QF_LRA)
    if mod is not None:
        print(mod)
        print(str(mod))
        # z3m = z3.Model()

        z3_model = Solver(name='z3').converter.convert()
        print(z3_model)
    else:
        print("unsat")

    """
    with Solver(logic=target_logic) as solver:
        solver.add_assertion(pysmt_fml)
        res = solver.solve()
        if res:
            model = [EqualsOrIff(k, solver.get_value(k)) for k in pysmt_vars]
            # TODO: map back to z3 model?
    """


def test():
    x, y = z3.Reals("x y")
    fml = z3.And(x > 3, y < 3)
    check_sat_with_pysmt(fml)


if __name__ == '__main__':
    test()
