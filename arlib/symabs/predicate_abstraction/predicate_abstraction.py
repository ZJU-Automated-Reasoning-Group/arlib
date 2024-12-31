"""
Predicate Abstraction

INPUT: a formula f and a set of predicates P = {p1, ..., pn}
OUTPUT: a formula g such that
        (1) g is a Boolean combination of P,
        (2) g is the strongest consequence of f.

     That is, f |= g and or any g' that is a Boolean combination of P, we have g |= g'.
"""

from typing import List

import z3
from z3 import BoolRef, Solver, unsat

from arlib.utils.z3_expr_utils import negate


def eval_predicates(m: z3.ModelRef, predicates: List[z3.BoolRef]):
    """
    The eval_preds function takes in a model m and a list of predicates preds.
    It returns the set of predicates that are true in m, or false if they are not.

    :param m:z3.ModelRef: Evaluate the predicates in the list of predicates
    :param predicates:List[z3.BoolRef]: Specify the set of predicates that we want to evaluate
    :return: A list of predicates that are true in the model m
    """
    res = []
    for p in predicates:
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


def predicate_abstraction(fml: z3.ExprRef, predicates: List[z3.ExprRef]) -> z3.ExprRef:
    """Compute the strongest necessary condition of fml that is the Boolean combination of preds

    Following CAV'06 paper "SMT Techniques for Fast Predicate Abstraction"
    (at least from my understanding...)

    TODO: indeed, the algorithm in the paper relies on the ``all-sat'' feature of MathSAT.
     So, the following code does not strictly follow the paper.
    """
    s = z3.Solver()
    s.add(fml)
    res = []
    while s.check() == z3.sat:
        m = s.model()
        # i.e., compute a prime/minimal implicant (using the agove prime_implicant function)
        projs = z3.And(eval_predicates(m, predicates))
        # projs = prime_implicant(projs, fml) # this is for further possible optimization
        # print(projs)
        res.append(projs)
        s.add(negate(projs))

    return z3.simplify(z3.Or(res))
