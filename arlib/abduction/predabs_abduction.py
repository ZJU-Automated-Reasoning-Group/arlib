# coding: utf-8
"""
Copyright 2021 rainoftime

Propositional abduction via predicate abstraction

Author:
    rainoftime@gmail.com
"""

from typing import List

import z3


def negate(f: z3.BoolRef) -> z3.BoolRef:
    """
    The negate function takes a z3.BoolRef and returns the negation of it.
    If f is already a negation, then it returns the argument of that negation.

    :param f:z3.BoolRef: Specify the formula that is to be negated
    :return: The negation of the input boolean expression
    """
    if z3.is_not(f):
        return f.arg(0)
    else:
        return z3.Not(f)


def eval_preds(m: z3.ModelRef, preds: List[z3.BoolRef]):
    """
    The eval_preds function takes in a model m and a list of predicates preds.
    It returns the set of predicates that are true in m, or false if they are not.

    :param m:z3.ModelRef: Evaluate the predicates in the list of predicates
    :param preds:List[z3.BoolRef]: Specify the set of predicates that we want to evaluate
    :return: A list of predicates that are true in the model m
    """
    res = []
    for p in preds:
        if z3.is_true(m.eval(p, True)):
            res.append(p)
        elif z3.is_false(m.eval(p, True)):
            res.append(negate(p))
        else:
            pass
    return res


def prime_implicant(ps: List[z3.ExprRef], e: z3.ExprRef):
    """TODO: this function may have flaws
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


def entail(a: z3.BoolRef, b: z3.BoolRef):
    """Check if a entails b (for testing whether the abduction algo. works)
    """
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(a, b)))
    return s.check() == z3.unsat


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
    res = []
    assert k > 0
    for i in range(k):
        s = z3.Solver()
        s.add(fml)
        s.set("random_seed", i)
        if s.check() == z3.sat:
            m = s.model()
            proj = z3.And(eval_preds(m, preds))
            print("implicant? ", entail(proj, fml))
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


def test_abduction():
    a, b, c, d = z3.Bools("a b c d")
    precond = z3.Or(a, b, z3.And(c, d))
    postcond = b
    res = abduction(precond, postcond, [a, c, d])
    print(res)
    if z3.is_false(res):
        print("cannot find the hypothesis using target_vars!")
        exit(0)
    # check if the algo. works
    print(entail(z3.And(res, precond), postcond))


test_abduction()