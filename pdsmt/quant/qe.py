# coding: utf-8
from typing import List

from z3 import *

"""
Implementation of the QE algorithm in the following paper:
 Quantifier elimination via lazy model enumeration, CAV??
"""


def negate(f: BoolRef) -> BoolRef:
    """
    negate, avoid double negation
    """
    if is_not(f):
        return f.arg(0)
    else:
        return Not(f)


def eval_preds(m: ModelRef, preds: List[BoolRef]):
    """
    Let m be a model of a formula phi
    preds be a set of predicates
    """
    res = []
    for p in preds:
        if is_true(m.eval(p)):
            res.append(p)
        elif is_false(m.eval(p)):
            res.append(negate(p))
        else:
            pass
    return res


def get_atoms(e: BoolRef):
    """
    Get all atomic predicates in a formula
    """
    s = set()

    def get_preds_(e):
        if e in s:
            return
        if is_not(e):
            s.add(e)
        if is_and(e) or is_or(e):
            for e_ in e.children():
                get_preds_(e_)
            return
        assert (is_bool(e))
        s.add(e)

    # convert to NNF and then look for preds
    ep = Tactic('nnf')(e).as_expr()
    get_preds_(ep)
    return s


def qelim_exists_lme(phi, qvars):
    """
    Existential Quantifier Elimination
    """
    s = z3.Solver()
    s.add(phi)
    res = []
    preds = get_atoms(phi)
    # NOTE: the tactic below only needs to handle conjunction of literals
    qe_for_conjunction = Tactic('qe2')  # or qe
    # similar to lazy DPLL(T)...
    while s.check() == sat:
        m = s.model()
        minterm = And(eval_preds(m, preds))
        proj = qe_for_conjunction(Exists(qvars, minterm)).as_expr()  # "forget" x in minterm
        res.append(proj)
        s.add(negate(proj))  # block the current one

    return simplify(Or(res))


def test_qe():
    x, y, z = BitVecs("x y z", 16)
    fml = And(Or(x > 2, x < y + 3), Or(x - z > 3, z < 10))  # x: 4, y: 1
    qf = qelim_exists_lme(fml, [x, y])
    print(qf)


if __name__ == "__main__":
    test_qe()
