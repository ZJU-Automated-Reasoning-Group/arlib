"""
Quantifier Elimination via Lazy Model Enumeration (LME-QE)

Implementation based on the paper:
"Quantifier elimination via lazy model enumeration" (CAV)

The algorithm eliminates existential quantifiers through iterative model enumeration
and projection. It works by:
1. Finding a model of the formula
2. Creating a minterm from the model
3. Projecting away quantified variables
4. Blocking the current solution
5. Repeating until no more models exist
"""

from typing import List

import z3

from arlib.utils.z3_expr_utils import negate


def eval_predicates(m: z3.ModelRef, preds: List[z3.ExprRef]) -> List[z3.ExprRef]:
    """Let m be a model of a formula phi preds be a set of predicates
    """
    res = []
    for p in preds:
        if z3.is_true(m.eval(p)):
            res.append(p)
        elif z3.is_false(m.eval(p)):
            res.append(negate(p))
        else:
            pass
    return res


def get_atoms(expr: z3.ExprRef) -> List[z3.ExprRef]:
    """Get all atomic predicates in a formula
    """
    s = set()

    def get_preds_(exp):
        if exp in s:
            return
        if z3.is_not(exp):
            s.add(exp)
        if z3.is_and(exp) or z3.is_or(exp):
            for e_ in exp.children():
                get_preds_(e_)
            return
        assert (z3.is_bool(exp))
        s.add(exp)

    # convert to NNF and then look for preds
    ep = z3.Tactic('nnf')(expr).as_expr()
    get_preds_(ep)
    return list(s)


def process_model(phi, qvars, preds, shared_models):
    """Worker function to process a single model"""
    s = z3.Solver()
    s.add(phi)

    # Block already found models
    for model in shared_models:
        s.add(negate(model))

    if s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        qe_for_conjunction = z3.Tactic('qe2')
        proj = qe_for_conjunction(z3.Exists(qvars, minterm)).as_expr()
        return proj
    return None


def qelim_exists_lme(phi, qvars):
    """
    Existential Quantifier Elimination
    """
    s = z3.Solver()
    s.add(phi)
    res = []
    preds = get_atoms(phi)
    # NOTE: the tactic below only needs to handle conjunction of literals
    qe_for_conjunction = z3.Tactic('qe2')  # or qe
    # similar to lazy DPLL(T)...
    while s.check() == z3.sat:
        m = s.model()
        # Create minterm from current mode
        minterm = z3.And(eval_predicates(m, preds))
        # Project away quantified variables
        proj = qe_for_conjunction(z3.Exists(qvars, minterm)).as_expr()  # "forget" x in minterm
        res.append(proj)
        # block the current one
        s.add(negate(proj))

    return z3.simplify(z3.Or(res))


def test_qe():
    # x, y, z = z3.BitVecs("x y z", 16)
    x, y, z = z3.Reals("x y z")
    fml = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))  # x: 4, y: 1
    qf = qelim_exists_lme(fml, [x, y])
    print(qf)


if __name__ == "__main__":
    test_qe()
