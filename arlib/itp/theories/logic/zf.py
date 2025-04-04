import arlib.itp as itp
import arlib.itp.smt as smt

ZFSet = smt.DeclareSort("ZFSet")
A, B, x, y, z = smt.Consts("A B x y z", ZFSet)
elem = smt.Function("elem", ZFSet, ZFSet, smt.BoolSort())
Class = ZFSet >> smt.BoolSort()
P, Q = smt.Consts("P Q", Class)
klass = itp.define("klass", [A], smt.Lambda([x], elem(x, A)))

zf_db = []


def slemma(thm, by=[], **kwargs):
    return itp.prove(thm, by=by + zf_db, **kwargs)


emp = smt.Const("emp", ZFSet)
emp_ax = itp.axiom(smt.ForAll([x], smt.Not(elem(x, emp))))

upair = smt.Function("upair", ZFSet, ZFSet, ZFSet)
upair_ax = itp.axiom(
    itp.QForAll([x, y, z], elem(z, upair(x, y)) == smt.Or(z == x, z == y))
)

ext_ax = itp.axiom(
    itp.QForAll([A, B], smt.ForAll([x], elem(x, A) == elem(x, B)) == (A == B))
)

sep = smt.Function("sep", ZFSet, Class, ZFSet)
sep_ax = itp.axiom(
    itp.QForAll([P, A, x], elem(x, sep(A, P)) == smt.And(P[x], elem(x, A)))
)

zf_db.extend([emp_ax, upair_ax, ext_ax, sep_ax])
le = itp.notation.le.define([A, B], itp.QForAll([x], elem(x, A), elem(x, B)))
