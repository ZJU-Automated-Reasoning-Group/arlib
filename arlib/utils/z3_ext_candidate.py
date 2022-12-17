# coding: utf-8
import z3

"""
- absolute_value_bv
- absolute_value_int
- ground_quantifier
- ground_quantifier_all
- ctx_simplify
"""


def absolute_value_bv(bv):
    """
    Based on: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
    Operation:
        Desired behavior is by definition (bv < 0) ? -bv : bv
        Now let mask := bv >> (bv.size() - 1)
        Note because of sign extension:
            bv >> (bv.size() - 1) == (bv < 0) ? -1 : 0
        Recall:
            -x == ~x + 1 => ~(x - 1) == -(x - 1) -1 == -x
            ~x == -1^x
             x ==  0^x
        now if bv < 0 then -bv == -1^(bv - 1) == mask ^ (bv + mask)
        else bv == 0^(bv + 0) == mask^(bv + mask)
        hence for all bv, absolute_value(bv) == mask ^ (bv + mask)
    """
    mask = bv >> (bv.size() - 1)
    return mask ^ (bv + mask)


def absolute_value_int(x):
    """
    Absolute value for integer encoding
    """
    return z3.If(x >= 0, x, -x)


def ground_quantifier(qexpr):
    """
    Seems this can only handle exists x . fml, or forall x.fml?
    """
    body = qexpr.body()
    var_list = list()
    for i in range(qexpr.num_vars()):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        var_list.append(vi)

    body = z3.substitute_vars(body, *var_list)
    return body, var_list


def ground_quantifier_all(qexpr):
    """
    Handle also Exists x . Forall y . Exists ..
    However, the following information is lost
    - order of the quantifiers
    - which variables are after which quantifiers
    """
    res = []
    exp = qexpr
    while True:
        exp, var_list = ground_quantifier(exp)
        res += var_list
        if not z3.is_quantifier(exp):
            break
    return exp, res


def subterms(t):
    seen = {}

    def subterms_rec(term):
        if z3.is_app(term):
            for ch in term.children():
                if ch in seen:
                    continue
                seen[ch] = True
                yield ch
                for sub in subterms_rec(ch):
                    yield sub

    return {s for s in subterms_rec(t)}


def are_equal(s, t1, t2):
    s.push()
    s.add(t1 != t2)
    r = s.check()
    s.pop()
    return r == z3.unsat


def ctx_simplify(slv, mdl, t):
    subs = subterms(t)
    values = {s: mdl.eval(s) for s in subs}
    values[t] = mdl.eval(t)

    def simplify_rec(term):
        m_subs = subterms(term)
        for s in m_subs:
            if s.sort().eq(term.sort()) and values[s].eq(values[term]) and are_equal(slv, s, term):
                return simplify_rec(s)
        chs = [simplify_rec(ch) for ch in term.children()]
        return term.decl()(chs)

    return simplify_rec(t)


def native_to_dnf(exp):
    # seems the result can be very verbose
    # ctx = exp.ctx_ref()
    # set_param("pp-min-alias-size", 1000000)
    # set_param("pp-max-depth", 1000000)
    to_cnf = z3.Then("simplify", "tseitin-cnf")
    to_dnf_helper = z3.Repeat(
        z3.OrElse("split-clause",
                  "skip"))
    to_dnf_tactic = z3.Then(to_cnf, to_dnf_helper)
    return to_dnf_tactic(exp).as_expr()


def test_quant():
    x, y, z = z3.Ints("x y z")
    fml = z3.Xor(z3.Or(x > 3, y + z < 100), z3.And(x < 100, y == 3), z3.Or(x + y < 100, y - z > 3))
    # qfml = z3.ForAll([x, y], fml)
    qfml2 = z3.ForAll(x, z3.Exists(y, fml))
    # print(ground_quantifier(qfml))
    # print(ground_quantifier(qfml2))
    print(ground_quantifier_all(qfml2))
    # print(find_all_uninterp_consts(fml))


def test_dnf():
    x, y, z = z3.Ints("x y z")
    fml = z3.Xor(z3.Or(x > 3, y + z < 100), z3.And(x < 100, y == 3), z3.Or(x + y < 100, y - z > 3))
    cnf_fml = z3.Then("simplify", "tseitin-cnf")(fml).as_expr()
    print(cnf_fml)
    # FIXME: this triggers an assertion error in prime_implicant
    # maybe caused by skelom constant(but the algo should be independent of the form)
    # print(exclusive_to_dnf(cnf_fml))

# test_quant()
