# coding: utf-8
from z3 import *


def entail(a, b):
    s = Solver()
    s.add(Not(Implies(a, b)))
    return s.check() == unsat


def get_atoms(e):
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

    ep = Tactic('nnf')(e).as_expr()
    get_preds_(ep)
    return s


def enumerate_literals(exp, atoms):
    res = []
    for atom in atoms:
        if entail(exp, atom):
            res.append(atom)
        elif entail(exp, Not(atom)):
            res.append(Not(atom))
    return res


def model_based_filtering(exp, atoms):
    """
    TODO
    """
    s = Solver()
    s.add(exp)
    while s.check() == sat:
        m = s.model()
        for atom in atoms:
            if is_false(m.eval(atom)):
                x = 1


def get_backbone(exp):
    atoms = get_atoms(exp)
    print(enumerate_literals(exp, atoms))
