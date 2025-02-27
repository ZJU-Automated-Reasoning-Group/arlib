# coding: utf-8
"""
Copyright 2021 rainoftime

Propositional abduction via predicate abstraction

Author:
    rainoftime@gmail.com
"""

from typing import List

import z3
from z3 import BoolRef, Solver, unsat

from arlib.utils.z3_expr_utils import negate


def eval_preds(m: z3.ModelRef, preds: List[z3.BoolRef]) -> List[z3.BoolRef]:
    """Evaluate predicates against a model and return their truth values.
    
    Args:
        m: Z3 model to evaluate against
        preds: List of predicates to evaluate
    Returns:
        List of predicates or their negations based on model evaluation
    """
    res = []
    for p in preds:
        val = m.eval(p, True)
        if z3.is_true(val):
            res.append(p)
        elif z3.is_false(val):
            res.append(negate(p))
    return res


def prime_implicant(ps: List[z3.ExprRef], e: z3.ExprRef):
    """TODO: this function may have flaws
    Prime implicant: an implicant that is not covered by any other implicant.
    """
    s = z3.Solver()
    # we want to find a subset ps' of ps such that /\ ps => e
    s.add(z3.Not(e))
    # holds temp bool vars for unsat core
    bs = []
    # map from temp vars to predicates
    btop = {}
    i = 0
    for p in ps:
        bp = z3.Bool("b" + str(i))
        btop[bp] = p
        bs.append(bp)
        s.add(z3.Implies(bp, p))
        i = i + 1
    # assert (s.check(bs) == unsat)
    # only take predicates in unsat core
    res = [btop[x] for x in s.unsat_core()]
    return res


def check_entailment(antecedent: BoolRef, consequent: BoolRef) -> bool:
    """Check if antecedent entails consequent.

    Args:
        antecedent: Formula that might entail
        consequent: Formula that might be entailed

    Returns:
        True if antecedent entails consequent, False otherwise
    """
    solver = Solver()
    solver.add(z3.Not(z3.Implies(antecedent, consequent)))
    return solver.check() == unsat


def predicate_abstraction(fml, preds):
    """Compute the strongest necessary condition of fml that is the Boolean combination of preds

    Following CAV'06 paper "SMT Techniques for Fast Predicate Abstraction"
    (at least from my understanding...)
    """
    s = z3.Solver()
    s.add(fml)
    res = []
    while s.check() == z3.sat:
        m = s.model()
        # TODO：a simple optimization is to use unsat core to reduce the size of proj
        # i.e., compute a prime/minimal implicant (using the agove prime_implicant function)
        projs = z3.And(eval_preds(m, preds))
        # projs = prime_implicant(projs, fml) # this is for further possible optimization
        # print(projs)
        res.append(projs)
        # print("implicant? ", entail(projs, And(s.assertions())))
        # print("implicant? ", entail(And(s.assertions()), projs))
        s.add(negate(projs))

    return z3.simplify(z3.Or(res))


def sample_k_implicants(fml, preds, k=-1):
    """
    """
    res = []
    assert k > 0
    for i in range(k):
        s = z3.Solver()
        s.add(fml)
        s.set("random_seed", i)
        if s.check() == z3.sat:
            m = s.model()
            proj = z3.And(eval_preds(m, preds))
            print("implicant? ", check_entailment(proj, fml))
            res.append(proj)
    print(res)
    return z3.simplify(z3.Or(res))


def ctx_simplify(e):
    return z3.Tactic('ctx-solver-simplify')(e).as_expr()


def abduction(precond, postcond, target_vars):
    """
    Given a set of premises Γ and a desired conclusion φ,
    abductive inference finds a simple explanation ψ such that
    (1) Γ ∧ ψ |= φ, and
    (2) ψ is consistent with known premises Γ.
    The key idea is that:  Γ ∧ ψ |= φ can be rewritten as ψ |= Γ -> φ.

    Then,
    1. compute the strongest necessary condition of Not(Γ -> φ) (via predicate abstraction)
    2. negate the result of the first step (i.e., the weakest sufficient condition of  Γ -> φ.

    target_vars: the variables to be used in the abductive hypothesis
    """
    fml = z3.Implies(precond, postcond)
    # pre_vars = get_vars(precond)
    # post_vars = get_vars(postcond)
    # target_vars = set.difference(set(pre_vars), set(post_vars))
    necessary = predicate_abstraction(z3.Not(fml), target_vars)
    sufficient = z3.simplify(z3.Not(necessary))
    # sufficient = ctx_simplify(Not(necsssary)) # use at your risk, as ctx_simplify can be slow
    # sufficient = sample_k_implicants(fml, target_vars, 10)
    return sufficient


def demo_abduction():
    a, b, c, d = z3.Bools("a b c d")
    precond = z3.Or(a, b, z3.And(c, d))
    postcond = b
    res = abduction(precond, postcond, [a, c, d])
    print(res)
    if z3.is_false(res):
        print("cannot find the hypothesis using target_vars!")
        exit(0)
    # check if the algo. works
    print(check_entailment(z3.And(res, precond), postcond))


if __name__ == "__main__":
    demo_abduction()
