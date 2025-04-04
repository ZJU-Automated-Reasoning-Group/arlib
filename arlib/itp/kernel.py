"""
The kernel hold core proof datatypes and core inference rules. By and large, all proofs must flow through this module.
"""

import arlib.itp as itp
import arlib.itp.smt as smt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import logging
from . import config

logger = logging.getLogger("itp")


@dataclass(frozen=True)
class Proof:
    """
    It is unlikely that users should be accessing the `Proof` constructor directly.
    This is not ironclad. If you really want the Proof constructor, I can't stop you.
    """

    thm: smt.BoolRef
    reason: list[Any]
    admit: bool = False

    def __post_init__(self):
        if self.admit and not config.admit_enabled:
            raise ValueError(
                self.thm, "was called with admit=True but config.admit_enabled=False"
            )

    def __hash__(self) -> int:
        return hash(self.thm)

    def _repr_html_(self):
        return "&#8870;" + repr(self.thm)

    def __repr__(self):
        return "|- " + repr(self.thm)

    def __call__(self, *args: smt.ExprRef):
        return instan(args, self)


"""
Proof_new = Proof.__new__

def sin_check(cls, thm, reason, admit=False, i_am_a_sinner=False):
    if admit and not config.admit_enabled:
        raise ValueError(
            thm, "was called with admit=True but config.admit_enabled=False"
        )
    if not i_am_a_sinner:
        raise ValueError("Proof is private. Use `itp.prove` or `itp.axiom`")
    return Proof_new(cls, thm, list(reason), admit)


Proof.__new__ = sin_check
"""


def is_proof(p: Proof) -> bool:
    return isinstance(p, Proof)


class LemmaError(Exception):
    pass


def prove(
    thm: smt.BoolRef,
    by: Proof | Iterable[Proof] = [],
    admit=False,
    timeout=1000,
    dump=False,
    solver=None,
) -> Proof:
    """Prove a theorem using a list of previously proved lemmas.

    In essence `prove(Implies(by, thm))`.

    :param thm: The theorem to prove.
    Args:
        thm (smt.BoolRef): The theorem to prove.
        by (list[Proof]): A list of previously proved lemmas.
        admit     (bool): If True, admit the theorem without proof.

    Returns:
        Proof: A proof object of thm

    >>> prove(smt.BoolVal(True))
    |- True
    >>> prove(smt.RealVal(1) >= smt.RealVal(0))
    |- 1 >= 0
    """
    if isinstance(by, Proof):
        by = [by]
    if admit:
        logger.warning("Admitting lemma {}".format(thm))
        return Proof(thm, list(by), admit=True)
    else:
        if solver is None:
            s = config.solver()  # type: ignore
        else:
            s = solver()
        s.set("timeout", timeout)
        for p in by:
            if not isinstance(p, Proof):
                raise LemmaError("In by reasons:", p, "is not a Proof object")
            s.add(p.thm)
        s.add(smt.Not(thm))
        if dump:
            print(s.sexpr())
        res = s.check()
        if res != smt.unsat:
            if res == smt.sat:
                raise LemmaError(thm, "Countermodel", s.model())
            raise LemmaError("prove", thm, res)
        else:
            return Proof(thm, list(by), False)


def axiom(thm: smt.BoolRef, by=["axiom"]) -> Proof:
    """Assert an axiom.

    Axioms are necessary and useful. But you must use great care.

    Args:
        thm: The axiom to assert.
        by: A python object explaining why the axiom should exist. Often a string explaining the axiom.
    """
    return Proof(thm, by)


@dataclass(frozen=True)
class Defn:
    """
    A record storing definition. It is useful to record definitions as special axioms because we often must unfold them.
    """

    name: str
    args: list[smt.ExprRef]
    body: smt.ExprRef
    ax: Proof


_datatypes = {}
defns: dict[smt.FuncDeclRef, Defn] = {}
"""
defn holds definitional axioms for function symbols.
"""
smt.FuncDeclRef.defn = property(lambda self: defns[self].ax)
smt.ExprRef.defn = property(lambda self: defns[self.decl()].ax)


def is_defined(x: smt.ExprRef) -> bool:
    """
    Determined if expression head is in definitions.
    """
    return smt.is_app(x) and x.decl() in defns


def fresh_const(q: smt.QuantifierRef):
    """Generate fresh constants of same sort as quantifier."""
    # .split("!") is to remove ugly multiple freshness from names
    return [
        smt.FreshConst(q.var_sort(i), prefix=q.var_name(i).split("!")[0])
        for i in range(q.num_vars())
    ]


def define(
    name: str, args: list[smt.ExprRef], body: smt.ExprRef, lift_lambda=False
) -> smt.FuncDeclRef:
    """
    Define a non recursive definition. Useful for shorthand and abstraction. Does not currently defend against ill formed definitions.
    TODO: Check for bad circularity, record dependencies

    Args:
        name: The name of the term to define.
        args: The arguments of the term.
        defn: The definition of the term.

    Returns:
        tuple[smt.FuncDeclRef, Proof]: A tuple of the defined term and the proof of the definition.
    """
    sorts = [arg.sort() for arg in args] + [body.sort()]
    f = smt.Function(name, *sorts)

    # TODO: This is getting too hairy for the kernel? Reassess. Maybe just a lambda flag? Autolift?
    if lift_lambda and isinstance(body, smt.QuantifierRef) and body.is_lambda():
        # It is worth it to avoid having lambdas in definition.
        vs = fresh_const(body)
        # print(vs, f(*args)[tuple(vs)])
        # print(smt.substitute_vars(body.body(), *vs))
        def_ax = axiom(
            smt.ForAll(
                args + vs,
                smt.Eq(
                    f(*args)[tuple(vs)], smt.substitute_vars(body.body(), *reversed(vs))
                ),
            ),
            by="definition",
        )
    elif len(args) == 0:
        def_ax = axiom(smt.Eq(f(), body), by="definition")
    else:
        def_ax = axiom(smt.ForAll(args, smt.Eq(f(*args), body)), by="definition")
    # assert f not in __sig or __sig[f].eq(   def_ax.thm)  # Check for redefinitions. This is kind of painful. Hmm.
    # Soft warning is more pleasant.
    defn = Defn(name, args, body, def_ax)
    if f not in defns or defns[f].ax.thm.eq(def_ax.thm):
        defns[f] = defn
    else:
        print("WARNING: Redefining function", f, "from", defns[f].ax, "to", def_ax.thm)
        defns[f] = defn
    if len(args) == 0:
        return f()  # Convenience
    else:
        return f


def define_fix(name: str, args: list[smt.ExprRef], retsort, fix_lam) -> smt.FuncDeclRef:
    """
    Define a recursive definition.
    """
    sorts = [arg.sort() for arg in args]
    sorts.append(retsort)
    f = smt.Function(name, *sorts)

    # wrapper to record calls
    calls = set()

    def record_f(*args):
        calls.add(args)
        return f(*args)

    defn = define(name, args, fix_lam(record_f))
    # TODO: check for well foundedness/termination, custom induction principle.
    return defn


def consider(x: smt.ExprRef) -> Proof:
    """
    The purpose of this is to seed the solver with interesting terms.
    Axiom schema. We may give a fresh name to any constant. An "anonymous" form of define.
    Pointing out the interesting terms is sometimes the essence of a proof.
    """
    return axiom(smt.Eq(smt.FreshConst(x.sort(), prefix="consider"), x))


"""
TODO: For better Lemma
def modus_n(n: int, ab: Proof, bc: Proof):
    ""
    Plug together two theorems of the form
    Implies(And(ps), b), Implies(And(qs, b, rs),  c)
    -----------
    Implies(And(qs, ps, rs), c)

    Useful for backwards chaining.

    
    ""
    assert (
        is_proof(ab)
        and is_proof(bc)
        and smt.is_implies(ab.thm)
        and smt.is_implies(bc.thm)
    )
    aa = ab.thm.arg(0)
    assert smt.is_and(aa)
    aa = aa.children()
    b = ab.thm.arg(1)
    bb = bc.thm.arg(0)
    assert smt.is_and(bb)
    bb = bb.children()
    assert bb[n].eq(b)
    c = bc.thm.arg(1)
    return axiom(smt.Implies(smt.And(*bb[:n], *aa, *bb[n + 1 :]), c), [ab, bc])
"""


def instan(ts: Sequence[smt.ExprRef], pf: Proof) -> Proof:
    """
    Instantiate a universally quantified formula.
    This is forall elimination
    """
    assert (
        is_proof(pf)
        and isinstance(pf.thm, smt.QuantifierRef)
        and pf.thm.is_forall()
        and len(ts) == pf.thm.num_vars()
    )

    return axiom(smt.substitute_vars(pf.thm.body(), *reversed(ts)), [pf])


def instan2(ts: Sequence[smt.ExprRef], thm: smt.BoolRef) -> Proof:
    """
    Instantiate a universally quantified formula
    `forall xs, P(xs) -> P(ts)`
    This is forall elimination
    """
    assert (
        isinstance(thm, smt.QuantifierRef)
        and thm.is_forall()
        and len(ts) == thm.num_vars()
    )

    return axiom(
        smt.Implies(thm, smt.substitute_vars(thm.body(), *reversed(ts))),
        ["forall_elim"],
    )


def forget(ts: Iterable[smt.ExprRef], pf: Proof) -> Proof:
    """
    "Forget" a term using existentials. This is existential introduction.
    This could be derived from forget2
    """
    # Hmm. I seem to have rarely been using this
    assert is_proof(pf)
    vs = [smt.FreshConst(t.sort()) for t in ts]
    return axiom(smt.Exists(vs, smt.substitute(pf.thm, *zip(ts, vs))), ["forget", pf])


def forget2(ts: Sequence[smt.ExprRef], thm: smt.QuantifierRef) -> Proof:
    """
    "Forget" a term using existentials. This is existential introduction.
    `P(ts) -> exists xs, P(xs)`
    `thm` is an existential formula, and `ts` are terms to substitute those variables with.
    forget easily follows.
    https://en.wikipedia.org/wiki/Existential_generalization
    """
    assert smt.is_quantifier(thm) and thm.is_exists() and len(ts) == thm.num_vars()
    return axiom(
        smt.Implies(smt.substitute_vars(thm.body(), *reversed(ts)), thm),
        ["exists_intro"],
    )


def einstan(thm: smt.QuantifierRef) -> tuple[list[smt.ExprRef], Proof]:
    """
    Skolemize an existential quantifier.
    `exists xs, P(xs) -> P(cs)` for fresh cs
    https://en.wikipedia.org/wiki/Existential_instantiation
    """
    # TODO: Hmm. Maybe we don't need to have a Proof? Lessen this to thm.
    assert smt.is_quantifier(thm) and thm.is_exists()

    skolems = fresh_const(thm)
    return skolems, axiom(
        smt.Implies(thm, smt.substitute_vars(thm.body(), *reversed(skolems))),
        ["einstan"],
    )


def skolem(pf: Proof) -> tuple[list[smt.ExprRef], Proof]:
    """
    Skolemize an existential quantifier.
    """
    # TODO: Hmm. Maybe we don't need to have a Proof? Lessen this to thm.
    assert is_proof(pf) and isinstance(pf.thm, smt.QuantifierRef) and pf.thm.is_exists()

    skolems = fresh_const(pf.thm)
    return skolems, axiom(
        smt.substitute_vars(pf.thm.body(), *reversed(skolems)), ["skolem", pf]
    )


def herb(thm: smt.QuantifierRef) -> tuple[list[smt.ExprRef], Proof]:
    """
    Herbrandize a theorem.
    It is sufficient to prove a theorem for fresh consts to prove a universal.
    Note: Perhaps lambdaized form is better? Return vars and lamda that could receive `|- P[vars]`
    """
    assert smt.is_quantifier(thm) and thm.is_forall()
    herbs = fresh_const(thm)
    return herbs, axiom(
        smt.Implies(smt.substitute_vars(thm.body(), *reversed(herbs)), thm),
        ["herband"],
    )


def beta_conv(lam: smt.QuantifierRef, *args) -> Proof:
    """
    Beta conversion for lambda calculus.
    """
    assert len(args) == lam.num_vars()
    assert smt.is_quantifier(lam) and lam.is_lambda()
    return axiom(smt.Eq(lam[args], smt.substitute_vars(lam.body(), *reversed(args))))


def induct_inductive(x: smt.DatatypeRef, P: smt.QuantifierRef) -> Proof:
    """Build a basic induction principle for an algebraic datatype"""
    DT = x.sort()
    assert isinstance(DT, smt.DatatypeSortRef)
    """assert (
        isisntance(P,QuantififerRef) and P.is_lambda()
    )  # TODO: Hmm. Actually it should just be arraysort"""
    hyps = []
    for i in range(DT.num_constructors()):
        constructor = DT.constructor(i)
        args = [
            smt.FreshConst(constructor.domain(j), prefix=DT.accessor(i, j).name())
            for j in range(constructor.arity())
        ]
        head = P(constructor(*args))
        body = [P(arg) for arg in args if arg.sort() == DT]
        if len(args) == 0:
            hyps.append(head)
        else:
            hyps.append(itp.QForAll(args, *body, head))
    conc = P(x)
    return axiom(smt.Implies(smt.And(hyps), conc), by="induction_axiom_schema")


def Inductive(name: str) -> smt.Datatype:
    """
    Declare datatypes with auto generated induction principles. Wrapper around z3.Datatype

    >>> Nat = Inductive("Nat")
    >>> Nat.declare("zero")
    >>> Nat.declare("succ", ("pred", Nat))
    >>> Nat = Nat.create()
    >>> Nat.succ(Nat.zero)
    succ(zero)
    """
    counter = 0
    n = name
    while n in _datatypes:
        counter += 1
        n = name + "!" + str(counter)
    name = n
    assert name not in _datatypes
    dt = smt.Datatype(name)
    oldcreate = dt.create

    def create():
        dt = oldcreate()
        # Sanity check no duplicate names. Causes confusion.
        names = set()
        for i in range(dt.num_constructors()):
            cons = dt.constructor(i)
            n = cons.name()
            if n in names:
                raise Exception("Duplicate constructor name", n)
            names.add(n)
            for j in range(cons.arity()):
                n = dt.accessor(i, j).name()
                if n in names:
                    raise Exception("Duplicate field name", n)
                names.add(n)
        itp.induct.register(dt, induct_inductive)
        _datatypes[name] = dt
        return dt

    dt.create = create
    return dt
