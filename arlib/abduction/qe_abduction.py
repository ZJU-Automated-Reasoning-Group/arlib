"""
Abduction via Quantifier Elimination
TBD: to check
"""

# This module provides methods for performing abduction via quantifier elimination.
# Quantifier elimination is a technique used in automated theorem proving and symbolic computation
# to eliminate quantifiers from a given formula, resulting in a quantifier-free formula.
# The main function in this module is `qe_abduce`, which takes a formula and returns a quantifier-free
# equivalent formula.

import z3


def get_vars(expr) -> set:
    """
    Extract all variables from a Z3 expression.
    
    Args:
        expr: Z3 expression

    Returns:
        set: Set of variables in the expression
    """
    vars_set = set()
    
    def collect(e):
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            vars_set.add(e)
        else:
            for child in e.children():
                collect(child)
    
    collect(expr)
    return vars_set
    

def quantifier_eliminiation(fml: z3.ExprRef) -> z3.ExprRef:
    """
    Perform quantifier elimination on the given formula.

    Args:
        fml (z3.ExprRef): The input formula to be processed.

    Returns:
        z3.ExprRef: The quantifier-free equivalent formula after performing quantifier elimination.
        The Tactic can be qe or qe2
    """
    return z3.Tactic("qe2")(fml).as_expr


def abduction(precond, postcond, target_vars) -> z3.ExprRef:
    """
    Given a set of premises Γ and a desired conclusion φ,
    abductive inference finds a simple explanation ψ such that
    (1) Γ ∧ ψ |= φ, and
    (2) ψ is consistent with known premises Γ.
    The key idea is that:  Γ ∧ ψ |= φ can be rewritten as ψ |= Γ -> φ.

    Then,
    1. compute the strongest necessary condition of Not(Γ -> φ) (via quantifier elimination)
    2. negate the result of the first step (i.e., the weakest sufficient condition of  Γ -> φ.

    Args:
        precond: The premise Γ (z3.ExprRef)
        postcond: The conclusion φ (z3.ExprRef)
        target_vars: List of variables to be used in the abductive hypothesis

    Returns:
        z3.ExprRef: The abductive hypothesis ψ
    """
    # Step 1: Form Γ -> φ
    implication = z3.Implies(precond, postcond)
    
    # Step 2: Negate the implication
    neg_implication = z3.Not(implication)
    
    # Step 3: Create existential quantifiers for non-target variables
    all_vars = get_vars(precond) | get_vars(postcond)
    quantified_vars = list(all_vars - set(target_vars))
    
    if quantified_vars:
        fml = z3.Exists(quantified_vars, neg_implication)
    else:
        fml = neg_implication
    
    # Step 4: Perform quantifier elimination
    eliminated = quantifier_eliminiation(fml)
    
    # Step 5: Negate the result to get the abductive hypothesis
    return z3.Not(eliminated)



 
