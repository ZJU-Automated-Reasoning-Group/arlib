"""
Backbone generation for SMT problems
"""
from typing import List
from z3 import *


def entail(a: z3.ExprRef, b: z3.ExprRef):
    """
    The entail function takes in two arguments, a and b. It then checks if the negation of
    a implies b. If it does not, then it returns true (i.e., entail(a,b) = True).
    Otherwise it returns false.

    :param a: Represent the knowledge base
    :param b: Check if the negation of a is entailed by b
    :return: `true` if `a` entails `b`, and `false` otherwise
    """
    s = Solver()
    s.add(Not(Implies(a, b)))
    return s.check() == unsat


def get_atoms(e: z3.ExprRef):
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


def enumerate_literals(exp: z3.ExprRef, atoms: List[z3.ExprRef]):
    """
    This function takes an expression and a list of atoms.
    It returns a list of all the literals that are entailed by the expression, 
    including both positive and negative literals.

    :param exp: The SMT formula F
    :param atoms: The set of atoms to be considered {p1,...,pn}
    :return: A list of literals that are entailed by F

    NOTE: If pi is a backbone (semantic consequence) of F,
    then pi evaluates to true under every model of F.
    """
    res = []
    for atom in atoms:
            s1 = Solver()
            s1.add(Not(Implies(exp, atom)))  # check for entailment
            if s1.check() == unsat:
                res.append(atom)
            else:
                m = s1.model()  # TODO: use m to prune other literals

            s2 = Solver()
            s2.add(Not(Implies(exp, Not(atom))))  # check for entailment
            if s2.check() == unsat:
                res.append(Not(atom))
            else:
                m = s2.model()  # TODO: use m to prune other literals

    return res


def get_backbone(exp):
    atoms = get_atoms(exp)
    print(enumerate_literals(exp, atoms))
