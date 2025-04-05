"""
Performing contextual simplification
"""
import z3


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
    """"""
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
