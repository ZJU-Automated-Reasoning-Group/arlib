"""
Abudciton via Quantifier Elimiination
"""


# This module provides methods for performing abduction via quantifier elimination.
# Quantifier elimination is a technique used in automated theorem proving and symbolic computation
# to eliminate quantifiers from a given formula, resulting in a quantifier-free formula.
# The main function in this module is `qe_abduce`, which takes a formula and returns a quantifier-free
# equivalent formula.

import z3

def quantifier_eliminiation(fml: z3.ExprRef):
    """
    Perform quantifier elimination on the given formula.

    Args:
        fml (z3.ExprRef): The input formula to be processed.

    Returns:
        z3.ExprRef: The quantifier-free equivalent formula after performing quantifier elimination.
        The Tactic can be qe or qe2
    """
    return z3.Tactic("qe2")(fml).as_expr


def abduction(precond, postcond, target_vars):
    """
    Given a set of premises Γ and a desired conclusion φ,
    abductive inference finds a simple explanation ψ such that
    (1) Γ ∧ ψ |= φ, and
    (2) ψ is consistent with known premises Γ.
    The key idea is that:  Γ ∧ ψ |= φ can be rewritten as ψ |= Γ -> φ.

    Then,
    1. compute the strongest necessary condition of Not(Γ -> φ) (via quantifier elimination)
    2. negate the result of the first step (i.e., the weakest sufficient condition of  Γ -> φ.

    target_vars: the variables to be used in the abductive hypothesis
    """
    fml = z3.Implies(precond, postcond)
    raise NotImplementedError()




