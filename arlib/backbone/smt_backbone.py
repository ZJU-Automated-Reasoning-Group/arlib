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

import random
from typing import List, Set
from z3 import *

from arlib.utils.z3_expr_utils import get_atoms
from itertools import combinations


def is_backbone_clause(fml: ExprRef, clause: List[ExprRef]) -> bool:
    """
    Checks if a given clause is a backbone clause.
    F |= (l1 ∨ ... ∨ ln) iff F ∧ ¬l1 ∧ ... ∧ ¬ln is UNSAT

    :param fml: The SMT formula F
    :param clause: List of literals forming the clause
    :return: True if the clause is a backbone clause
    """
    solver = Solver()
    solver.add(fml)
    # Add negation of all literals in the clause
    for lit in clause:
        solver.add(Not(lit))
    return solver.check() == unsat


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


def get_backbone_literals(fml: z3.ExprRef, atoms: List[z3.ExprRef]):
    """
    This function takes an expression and a list of atoms.
    It returns a list of all the literals that are entailed by the expression, 
    including both positive and negative literals.

    :param fml: The SMT formula F
    :param atoms: The set of atoms to be considered {p1,...,pn}
    :return: A list of literals that are entailed by F

    NOTE: If pi is a backbone (semantic consequence) of F,
    then pi evaluates to true under every model of F.
    """
    res = []
    solver = Solver()
    solver.add(fml)

    for atom in atoms:
        # Check if the positive literal is a backbone
        solver.push()
        solver.add(Not(atom))
        if solver.check() == unsat:
            res.append(atom)
        solver.pop()

        # Check if the negative literal is a backbone
        solver.push()
        solver.add(atom)
        if solver.check() == unsat:
            res.append(Not(atom))
        solver.pop()


def enumerate_backbone_clauses(fml: ExprRef, atoms: List[ExprRef], max_length: int = None) -> List[List[ExprRef]]:
    """
    Enumerates all backbone clauses up to the specified maximum length.
    Uses lazy model enumeration to prune the search space.

    :param fml: The SMT formula F
    :param atoms: The set of atoms to be considered {p1,...,pn}
    :param max_length: Maximum length of clauses to consider
    :return: List of backbone clauses
    """
    if max_length is None:
        max_length = len(atoms)

    # Generate all possible literals (positive and negative)
    literals = []
    for atom in atoms:
        literals.extend([atom, Not(atom)])

    backbone_clauses = []
    solver = Solver()
    solver.add(fml)

    # Try clauses of increasing length
    for length in range(1, max_length + 1):
        for clause_lits in combinations(literals, length):
            # Skip if any subset is already a backbone clause (optimization)
            if any(all(lit in clause_lits for lit in existing_clause)
                   for existing_clause in backbone_clauses):
                continue

            if is_backbone_clause(fml, list(clause_lits)):
                backbone_clauses.append(list(clause_lits))

    return backbone_clauses


def sample_backbone_clauses(fml: ExprRef, atoms: List[ExprRef],
                            num_samples: int, max_length: int = None) -> List[List[ExprRef]]:
    """
    Samples a specified number of backbone clauses randomly.

    :param fml: The SMT formula F
    :param atoms: The set of atoms to be considered {p1,...,pn}
    :param num_samples: Number of backbone clauses to sample
    :param max_length: Maximum length of clauses to consider
    :return: List of sampled backbone clauses
    """
    if max_length is None:
        max_length = len(atoms)

    literals = []
    for atom in atoms:
        literals.extend([atom, Not(atom)])

    samples = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(samples) < num_samples and attempts < max_attempts:
        length = random.randint(1, max_length)
        clause_lits = random.sample(literals, length)

        if is_backbone_clause(fml, clause_lits):
            samples.append(clause_lits)
        attempts += 1

    return samples
