# coding: utf8
from typing import List

import z3
from z3.z3util import get_vars

'''
Useful APIs 
- skolemize
- compact_check
- negate
- eval_preds
- get_variables
- get_atoms
- entail
- is_equivalent
- to_smtlib
- is_expr_var
- is_expr_val
- is_term
- get_models
- to_dnf
- exclusive_to_dnf
'''


def skolemize(exp: z3.ExprRef) -> z3.ExprRef:
    """Skolemize a formula (important for handling quantified formulas)"""
    g = z3.Goal()
    g.add(exp)
    t = z3.Tactic('snf')
    res = t(g)
    return res.as_expr()


def compact_check_misc(precond, cnt_list, res_label):
    """
    TODO: In our settings, as long as there is one unsat, we can stop
      However, this algorithm stops until "all the remaining ones are UNSAT (which can have one of more instances)"
    """
    f = z3.BoolVal(False)
    for i in range(len(res_label)):
        if res_label[i] == 2:
            f = z3.Or(f, cnt_list[i])
    if z3.is_false(f):
        return

    # sol = z3.SolverFor("QF_BV")
    sol = z3.Solver()
    g = z3.And(precond, f)
    sol.add(g)
    s_res = sol.check()
    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif s_res == z3.sat:
        m = sol.model()
        # models.append(m)  # counterexample
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], True)):
                res_label[i] = 1
    else:
        return
    compact_check_misc(precond, cnt_list, res_label)


def compact_check(precond: z3.BoolRef, cnt_list: List[z3.BoolRef]) -> List[int]:
    """
    Given a precond G and a set of cnts: f1, f2,..., fn
    Decis if the following cnts are satisfiable:
           And(G, f1), And(G, f2), ..., And(G, fn)
    Examples:
    >>> from z3 import *
    >>> x, y = Reals('x y')
    >>> pre = x > 100
    >>> f1 = x > 0; f2 = And(x > y, x < y); f3 = x < 3
    >>> cnts = [f1, f2, f3]
    >>> assert compact_check(pre, cnts) == [1, 0, 0]
    """
    res = []
    for _ in cnt_list:
        res.append(2)
    compact_check_misc(precond, cnt_list, res)
    return res


def negate(f: z3.BoolRef) -> z3.BoolRef:
    """
    negate, avoid double negation
    """
    if z3.is_not(f):
        return f.arg(0)
    else:
        return z3.Not(f)


def eval_preds(m: z3.ModelRef, preds: List[z3.BoolRef]):
    """
    Let m be a model of a formula phi
    preds be a set of predicates
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


def get_variables(exp: z3.ExprRef):
    """
    Get variables
    """
    return get_vars(exp)


def get_atoms(expr: z3.BoolRef):
    """
    Get all atomic predicates in a formula
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
    return s


def entail(a: z3.BoolRef, b: z3.BoolRef) -> bool:
    """
    Check if a entails b
    Examples:
    >>> from z3 import *
    >>> assert entail(Bool('z'), Bool('z'))
    >>> x, y = Ints("x y")
    >>> assert entail(x < 5, x < 100)
    """
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(a, b)))
    return s.check() == z3.unsat


def is_equivalent(a: z3.BoolRef, b: z3.BoolRef) -> bool:
    """
    Check if a and b are equivalent
    Examples:
    >>> from z3 import *
    >>> assert is_equivalent(Bool('z'), Bool('z'))
    >>> x, y = Ints("x y")
    >>> assert is_equivalent(x > 2, x >= 3)
    """
    s = z3.Solver()
    # s.set("timeout", 5000)
    s.add(a != b)
    return s.check() == z3.unsat


def to_smtlib2(expr: z3.BoolRef) -> str:
    """"
    To SMT-LIB2 string
    """
    s = z3.Solver()
    s.add(expr)
    return s.to_smt2()


def is_expr_var(a) -> bool:
    """
    Check if a is a variable. E.g. x is a var but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert is_expr_var(Int('7'))
    >>> assert not is_expr_var(IntVal('7'))
    >>> assert is_expr_var(Bool('y'))
    >>> assert not is_expr_var(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_var(LOnOff)
    >>> assert not is_expr_var(On)
    >>> assert is_expr_var(Block)
    >>> assert is_expr_var(SafetyInjection)
    """

    return z3.is_const(a) and a.decl().kind() == z3.Z3_OP_UNINTERPRETED


def is_expr_val(a) -> bool:
    """
    Check if the input formula is a value. E.g. 3 is a value but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert not is_expr_val(Int('7'))
    >>> assert is_expr_val(IntVal('7'))
    >>> assert not is_expr_val(Bool('y'))
    >>> assert not is_expr_val(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_val(LOnOff)
    >>> assert is_expr_val(On)
    >>> assert not is_expr_val(Block)
    >>> assert not is_expr_val(SafetyInjection)
    """
    return z3.is_const(a) and a.decl().kind() != z3.Z3_OP_UNINTERPRETED


def is_term(a) -> bool:
    """
    Check if the input formula is a term. In FOL, terms are
    defined as term := const | var | f(t1,...,tn) where ti are terms.
    Examples:
    >>> from z3 import *
    >>> assert is_term(Bool('x'))
    >>> assert not is_term(And(Bool('x'),Bool('y')))
    >>> assert not is_term(And(Bool('x'),Not(Bool('y'))))
    >>> assert is_term(IntVal(3))
    >>> assert is_term(Int('x'))
    >>> assert is_term(Int('x') + Int('y'))
    >>> assert not is_term(Int('x') + Int('y') > 3)
    >>> assert not is_term(And(Int('x')==0,Int('y')==3))
    >>> assert not is_term(Int('x')==0)
    >>> assert not is_term(3)
    >>> assert not is_term(Bool('x') == (Int('y')==Int('z')))
    """

    if not z3.is_expr(a):
        return False
    if z3.is_const(a):  # covers both const value and var
        return True
    else:  # covers f(t1,..,tn)
        return not z3.is_bool(a) and all(is_term(c) for c in a.children())


CONNECTIVE_OPS = [z3.Z3_OP_NOT, z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_IMPLIES,
                  z3.Z3_OP_IFF, z3.Z3_OP_ITE, z3.Z3_OP_XOR]


def is_atom(a) -> bool:
    """
    Check if the input formula is an atom. In FOL, atoms are
    defined as atom := t1 = t2 | R(t1,..,tn) where ti are terms.
    In addition, this function also allows Bool variable to
    be terms (in propositional logic, a bool variable is considered term)
    """
    if not z3.is_bool(a):
        return False

    if is_expr_val(a):
        return False

    if is_expr_var(a):
        return True

    return z3.is_app(a) and a.decl().kind() not in CONNECTIVE_OPS and all(is_term(c) for c in a.children())


def is_pos_lit(a) -> bool:
    """
    Check if the input formula is a positive literal,  i.e. an atom
    >>> is_pos_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_atom(a)


def is_neg_lit(a) -> bool:
    """
    Check if the input formula is a negative literal
    EXAMPLES:
    >>> from z3 import *
    >>> is_term(3)
    False
    >>> is_neg_lit(Not(Bool('x')))
    True
    >>> is_neg_lit(Not(BoolVal(False)))
    False
    >>> is_neg_lit(BoolVal(True))
    False
    >>> is_neg_lit(BoolVal(False))
    False
    >>> is_neg_lit(Not(Int('x') + Int('y') > 3))
    True
    >>> is_neg_lit(Not(Bool('x') == BoolVal(True)))
    True
    >>> is_neg_lit(Not(Int('x') == 3))
    True
    >>> is_neg_lit(Not(BoolVal(True)))
    False
    """
    return z3.is_not(a) and is_pos_lit(a.children()[0])


def is_lit(a) -> bool:
    """
    Check if the input formula is a negative literal
    >>> is_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_pos_lit(a) or is_neg_lit(a)


def get_models(f: z3.BoolRef, k: int):
    """
    Returns the first k models satisfiying f.
    If f is not satisfiable, returns False.
    If f cannot be solved, returns None
    If f is satisfiable, returns the first k models
    Note that if f is a tautology, e.g. True, then the result is []
    Based on http://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation
    EXAMPLES:
    >>> from z3 import *
    >>> x, y = Ints('x y')
    >>> len(get_models(And(0<=x,x <= 4),k=11))
    5
    >>> get_models(And(0<=x**y,x <= 1),k=2) is None
    True
    >>> get_models(And(0<=x,x <= -1),k=2)
    False
    >>> len(get_models(x+y==7,5))
    5
    >>> len(get_models(And(x<=5,x>=1),7))
    5
    >>> get_models(And(x<=0,x>=5),7)
    False
    >>> x = Bool('x')
    >>> get_models(And(x,Not(x)),k=1)
    False
    >>> get_models(Implies(x,x),k=1)
    []
    """

    assert z3.is_expr(f), f
    assert k >= 1, k

    s = z3.Solver()
    s.add(f)
    models = []
    i = 0
    while s.check() == z3.sat and i < k:
        i = i + 1
        m = s.model()

        if not m:  # if m == []
            break
        models.append(m)
        # create new constraint to block the current model
        block = z3.Not(z3.And([var() == m[var] for var in m]))
        s.add(block)

    if s.check() == z3.unknown:
        return None
    elif s.check() == z3.unsat and i == 0:
        return False
    else:
        return models


def prime_implicant(ps: List[z3.ExprRef], expr: z3.ExprRef):
    """
    TODO: this function may have flaws (need to figure why the assertion below fails
    """
    s = z3.Solver()
    # we want to find a subset ps' of ps such that /\ ps => e
    s.add(z3.Not(expr))
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
    assert (s.check(bs) == z3.unsat)
    # only take predicates in unsat core
    res = [btop[x] for x in s.unsat_core()]
    return res


def big_and(exp_list):
    if not exp_list:
        return True
    if len(exp_list) == 1:
        return exp_list[0]
    return z3.And(*exp_list)


def big_or(ll):
    if not ll:
        return False
    if len(ll) == 1:
        return ll[0]
    return z3.Or(*ll)


def to_dnf(phi: z3.BoolRef, maxlen=None):
    s = z3.Solver()
    s.add(phi)
    # get all predicates in phi
    preds = get_atoms(phi)
    # disjuncts
    # phiprime = phi
    res = []

    while s.check() == z3.sat:
        # get model
        m = s.model()
        # evaluate model --> get disjunct
        dd = eval_preds(m, list(preds))
        # print("size before", len(d))

        # get prime implicant of disjunct
        dd = prime_implicant(dd, phi)
        # print("size after", len(d))

        # asserthe negation of disjunct to avoid getting it again
        s.add(z3.Not(big_and(dd)))
        # phiprime = And(phiprime,Not(bigAnd(d)))

        res = res + [big_and(dd)]
        if maxlen is not None:
            if len(res) > maxlen:
                return []
    # NOTE: sanity checking code, ensures DNF is equivalent to phi
    # resphi = Or(*res)
    # assert is_equivalent(resphi, phi)

    # return dnf as list
    return res


def exclusive_to_dnf(phi: z3.BoolRef, maxlen=None):
    s = z3.Solver()
    s.add(phi)
    # get all predicates in phi
    preds = get_atoms(phi)
    # disjuncts
    phiprime = phi
    res = []
    while s.check() == z3.sat:
        # get model
        m = s.model()
        # evaluate model --> get disjunct
        dd = eval_preds(m, list(preds))
        # print("size before", len(d))
        # get prime implicant of disjunct
        dd = prime_implicant(dd, phiprime)
        # print("size after", len(d))
        # asserthe negation of disjunct to avoid getting it again
        s.add(z3.Not(big_and(dd)))
        phiprime = z3.And(phiprime, z3.Not(big_and(dd)))
        new_entry = big_and(dd)
        res = res + [new_entry]
        if maxlen is not None:
            if len(res) > maxlen:
                return []

    # NOTE: sanity checking code, ensures DNF is equivalent to phi
    # resphi = Or(*res)
    # assert is_equivalent(resphi, phi)
    return res
    # return dnf as list


class FormulaInfo:
    def __init__(self, fml):
        self.formula = fml
        self.has_quantifier = self.has_quantifier()
        self.logic = self.get_logic()

    def apply_probe(self, name):
        g = z3.Goal()
        g.add(self.formula)
        p = z3.Probe(name)
        return p(g)

    def has_quantifier(self):
        return self.apply_probe('has-quantifiers')

    def logic_has_bv(self):
        return "BV" in self.logic

    def get_logic(self):
        """
        TODO: how about string, array, and FP?
        """
        try:
            if not self.has_quantifier:
                if self.apply_probe("is-propositional"):
                    return "QF_UF"
                elif self.apply_probe("is-qfbv"):
                    return "QF_BV"
                elif self.apply_probe("is-qfaufbv"):
                    return "QF_AUFBV"
                elif self.apply_probe("is-qflia"):
                    return "QF_LIA"
                elif self.apply_probe("is-quauflia"):
                    return "QF_AUFLIA"
                elif self.apply_probe("is-qflra"):
                    return "QF_LRA"
                elif self.apply_probe("is-qflira"):
                    return "QF_LIRA"
                elif self.apply_probe("is-qfnia"):
                    return "QF_NIA"
                elif self.apply_probe("is-qfnra"):
                    return "QF_NRA"
                elif self.apply_probe("is-qfufnra"):
                    return "QF_UFNRA"
                else:
                    return "ALL"
            else:
                if self.apply_probe("is-lia"):
                    return "LIA"
                elif self.apply_probe("is-lra"):
                    return "LRA"
                elif self.apply_probe("is-lira"):
                    return "LIRA"
                elif self.apply_probe("is-nia"):
                    return "NIA"
                elif self.apply_probe("is-nra"):
                    return "NRA"
                elif self.apply_probe("is-nira"):
                    return "NIRA"
                else:
                    return "ALL"
        except Exception as ex:
            print(ex)
            return "ALL"


if __name__ == '__main__':
    import doctest

    doctest.testmod()
