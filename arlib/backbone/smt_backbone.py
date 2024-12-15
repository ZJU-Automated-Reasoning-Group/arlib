"""
Backbone clauses generation via lazy model enumeration
  Consider a set of predicates S = {p1, ..., pn}
  A clause C is the disjunction of a subset of literals, each of the literal
   uses predicates from S (e.g., a literal can be p1, not p1, ....)

  If F |= C, then we say C is a backbone clause of F w.r.t. S.
   F |= C means Not(F -> C) is unsat, which means F and (not C) is unsat.

   For example, let C be "p1 or p2 or p3". To check whether C is a backbone clause of F or not,
   We need to show that "F and (not p1) and (not p2) and (not p3)" is unsatisfiable.
   Then, maybe we can reduce to something like "unsat core enumeration"?

   (FIXME: how to tell an SMT solver that "the unsat core must contain F")

 - Generation (an arbitrary one?), or sampling (a set of backbones), or enumeration (all)?
 - Restrict the length of the clause?
 - Where is S from?

TODO:
  - Combination with predicate abstraction
    + Predicate abstraction aims to compute the strongest consequence of F, which is expressible as a
    Boolean combination of S. (e.g., p1 or (not p2))
    + Backbone clauses enumeration aims to compute all blabla....?
  - Comparison with SAT backbone
    + SAT backbone:
    + ...?
  - Comparison with Unsat core enumeration? (Maybe we can reuse some existing algorithms)
"""

from typing import List
from z3 import *

from arlib.utils.z3_expr_utils import get_atoms


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
    solver = Solver()

    for atom in atoms:
        # Check if the positive literal is a backbone
        solver.push()
        solver.add(exp)
        solver.add(Not(atom))
        if solver.check() == unsat:
            res.append(atom)
        solver.pop()

        # Check if the negative literal is a backbone
        solver.push()
        solver.add(exp)
        solver.add(atom)
        if solver.check() == unsat:
            res.append(Not(atom))
        solver.pop()


def get_backbone(exp):
    atoms = get_atoms(exp)
    print(enumerate_literals(exp, atoms))
