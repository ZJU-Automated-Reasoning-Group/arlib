"""
Backbone generation for SMT problems
"""
from z3 import *


def entail(a, b):
    """
    The entail function takes in two arguments, a and b. It then checks if the negation of
    a implies b. If it does not, then it returns true (i.e., entail(a,b) = True). Otherwise
    it returns false.

    :param a: Represent the knowledge base
    :param b: Check if the negation of a is entailed by b
    :return: `true` if `a` entails `b`, and `false` otherwise
    """
    s = Solver()
    s.add(Not(Implies(a, b)))
    return s.check() == unsat


def get_atoms(e):
    """
    The get_atoms function takes a Z3 expression as input and returns the set of all
    atomic predicates that appear in the expression.

    :param e: Pass the expression to be converted
    :return: The set of atoms in a formula
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

    ep = Tactic('nnf')(e).as_expr()
    get_preds_(ep)
    return s


def enumerate_literals(exp, atoms):
    """
    The enumerate_literals function takes an expression and a list of atoms.
    It returns a list of all the literals that are entailed by the expression, including both positive and negative literals.

    :param exp: The expression
    :param atoms: The set of atoms to be considered
    :return: A list of literals that are entailed by the expression
    """
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
