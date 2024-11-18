# coding: utf-8
import z3
from pysmt.oracles import get_logic
from pysmt.shortcuts import EqualsOrIff
from pysmt.shortcuts import Symbol, And
from pysmt.typing import INT, REAL, BVType, BOOL
from pysmt.shortcuts import Not, Solver


# NOTE: both pysmt and z3 have a class "Solver"
# logger = logging.getLogger(__name__)

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


def convert(zf: z3.ExprRef):
    """
    FIXME: if we do not call "pysmt_vars = ...", z3 will report naming warning..
    """
    zvs = z3.z3util.get_vars(zf)
    pysmt_vars = to_pysmt_vars(zvs)
    z3s = Solver(name='z3')
    pysmt_fml = z3s.converter.back(zf)
    return pysmt_vars, pysmt_fml


def all_smt_with_pysmt(fml, keys, bound):
    """
    Sample k models
    """
    z3fml = z3.And(fml)
    pysmt_var_keys, pysmt_fml = convert(z3fml)
    target_logic = get_logic(pysmt_fml)

    # print("Target Logic: %s" % target_logic)

    with Solver(logic=target_logic) as solver:
        solver.add_assertion(pysmt_fml)
        iteration = 0
        while solver.solve():
            partial_model = [EqualsOrIff(k, solver.get_value(k)) for k in pysmt_var_keys]
            print(partial_model)
            solver.add_assertion(Not(And(partial_model)))
            iteration += 1
            if iteration >= bound: break
