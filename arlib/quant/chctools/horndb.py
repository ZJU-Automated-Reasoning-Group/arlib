import io
import sys

import pysmt.environment  # type: ignore
import pysmt.solvers.z3 as pyz3  # type: ignore
import z3  # type: ignore
from pysmt.smtlib.parser import SmtLibZ3Parser, Tokenizer  # type: ignore


def ground_quantifier(qexpr):
    body = qexpr.body()

    var_list = list()
    for i in reversed(range(qexpr.num_vars())):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        var_list.append(vi)

    body = z3.substitute_vars(body, *var_list)
    return body, var_list


def find_all_uninterp_consts(formula, res):
    if z3.is_quantifier(formula):
        formula = formula.body()

    worklist = []
    if z3.is_implies(formula):
        worklist.append(formula.arg(1))
        arg0 = formula.arg(0)
        if z3.is_and(arg0):
            worklist.extend(arg0.children())
        else:
            worklist.append(arg0)
    else:
        worklist.append(formula)

    for t in worklist:
        if z3.is_app(t) and t.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            res.append(t.decl())


class HornRule(object):
    def __init__(self, formula):
        self._ctx = formula.ctx
        self._formula = formula
        self._head = None
        self._body = []
        self._uninterp_sz = 0
        self._bound_constants = []

        self._update()

    def _update(self):
        if not self.has_formula():
            return

        rels = list()
        find_all_uninterp_consts(self._formula, rels)
        self._rels = frozenset(rels)
        body = self._formula
        if z3.is_quantifier(body):
            body, self._bound_constants = ground_quantifier(body)

        if z3.is_implies(body):
            self._head = body.arg(1)
            body = body.arg(0)
            if z3.is_and(body):
                body = body.children()
            else:
                body = [body]
        else:
            self._head = body
            body = []

        # remove all true constants
        body = [x for x in body if not z3.is_true(x)]

        if len(body) > 0:
            self._body = body

        for i in range(len(body)):
            f = body[i]
            if z3.is_app(f) and f.decl() in self._rels:
                self._uninterp_sz += 1
            else:
                break

        # reset _formula, it can be re-computed using mk_formula()
        # this ensures that any simplifications that are done during _update() are
        # also reflected in the formula view
        self._formula = None
        assert self._head is not None

    def __str__(self):
        return str(self._formula)

    def __repr__(self):
        return repr(self._formula)

    def used_rels(self):
        return self._rels

    def is_query(self):
        return z3.is_false(self._head)

    def is_simple_query(self):
        """Returns true if query is a simple.

        A simple query is an application of an uninterpreted predicate

        """
        if not self.is_query():
            return False

        if self.uninterp_size() != 1:
            return False

        predicate = self.body()[0]

        if predicate.num_args() > 0:
            return False

        _body = self.body()[1:]
        if len(_body) == 0:
            return True

        if len(_body) == 1:
            return z3.is_true(_body[0])

        _body = z3.simplify(z3.And(*_body, self._ctx))
        return z3.is_true(_body)

    # based on the following inference
    #
    # forall v :: (expr ==> false)
    #
    # equivalent to
    #
    # forall v:: ( expr ==> q ) && forall v :: ( q ==> false )
    #
    def split_query(self):
        """Split query if it is not simple into a query and a rule"""

        assert self.is_query()
        if self.is_simple_query():
            return self, None

        q = z3.Bool("simple!!query", self._ctx)
        query = HornRule(z3.Implies(q, False))
        if self._bound_constants:
            rule = HornRule(
                z3.ForAll(
                    self._bound_constants,
                    z3.Implies(z3.And(*self.body(), self._ctx), q),
                )
            )
        else:
            rule = HornRule(z3.Implies(z3.And(*self.body(), self._ctx), q))
        return query, rule

    def is_fact(self):
        return self._uninterp_sz == 0

    def is_linear(self):
        return self._uninterp_sz <= 1

    def to_ast(self):
        return self._formula

    def head(self):
        return self._head

    def body(self):
        return self._body

    def uninterp_size(self):
        return self._uninterp_sz

    def has_formula(self):
        return self._formula is not None

    def get_formula(self):
        return self._formula

    def mk_formula(self):
        f = self._body
        if len(f) == 0:
            f = z3.BoolVal(True, self._ctx)
        elif len(f) == 1:
            f = f[0]
        else:
            f = z3.And(f, self._ctx)
        f = z3.Implies(f, self._head)

        if len(self._bound_constants) > 0:
            f = z3.ForAll(self._bound_constants, f)
        self._formula = f
        return self._formula

    def mk_query(self):
        assert self.is_query()
        assert len(self.body()) > 0
        _body = self.body()
        if self.is_simple_query():
            return _body[0]

        if len(_body) == 1:
            f = _body[0]
        else:
            f = z3.And(_body, self._ctx)
        if len(self._bound_constants) > 0:
            f = z3.Exists(self._bound_constants, f)
        return f

    def get_ctx(self):
        return self._ctx


class HornRelation(object):
    def __init__(self, fdecl, env=None):
        self._fdecl = fdecl
        self._sig = []
        self._pysmt_sig = []
        self._lemma_parser = None
        if env is not None:
            self._env = env
        else:
            self._env = pysmt.environment.get_env()
        self._update()

    def _update(self):
        self._sig = []
        for i in range(self._fdecl.arity()):
            name = self._mk_arg_name(i)
            sort = self._fdecl.domain(i)
            self._sig.append(z3.Const(name, sort))

        # compute pysmt version of the signature

        mgr = self._env.formula_manager
        converter = pyz3.Z3Converter(self._env, self.get_ctx())
        # noinspection PyProtectedMember
        self._pysmt_sig = [
            mgr.Symbol(v.decl().name(), converter._z3_to_type(v.sort()))
            for v in self._sig
        ]

    def _mk_arg_name(self, i):
        # can be arbitrary convenient name
        return "{}_{}_n".format(self.name(), i)

    def _mk_lemma_arg_name(self, i):
        # must match name used in the lemma
        return "{}_{}_n".format(self.name(), i)

    def name(self):
        return str(self._fdecl.name())

    def __str__(self):
        return repr(self)

    def __repr__(self):
        import io

        out = io.StringIO()
        out.write(str(self.name()))
        out.write("(")
        for v in self._pysmt_sig:
            out.write(str(v))
            out.write(", ")
        out.write(")")
        return out.getvalue()

    def _mk_lemma_parser(self):
        if self._lemma_parser is not None:
            return
        self._lemma_parser = SmtLibZ3Parser()
        # register symbols that are expected to appear in the lemma
        for i, symbol in enumerate(self._pysmt_sig):
            name = self._mk_lemma_arg_name(i)
            self._lemma_parser.cache.bind(name, symbol)

    def pysmt_parse_lemma(self, lemma):
        self._mk_lemma_parser()
        tokens = Tokenizer(lemma, interactive=False)
        return self._lemma_parser.get_expression(tokens)

    def get_ctx(self):
        return self._fdecl.ctx


class HornClauseDb(object):
    def __init__(self, name="horn", simplify_queries=True, ctx=z3.main_ctx(), env=None):
        self._ctx = ctx
        self._name = name
        self._rules = []
        self._queries = []
        self._rels_set = frozenset()
        self._rels = dict()
        self._sealed = True
        self._fp = None
        self._env = env
        self._simple_query = simplify_queries

    def add_rule(self, horn_rule):
        assert self._ctx == horn_rule.get_ctx()
        self._sealed = False
        if horn_rule.is_query():
            if self._simple_query and not horn_rule.is_simple_query():
                query, rule = horn_rule.split_query()
                self._rules.append(rule)
                self._queries.append(query)
            else:
                self._queries.append(horn_rule)
        else:
            self._rules.append(horn_rule)

    def get_rels(self):
        self.seal()
        return self._rels_set

    def has_rel(self, rel_name):
        return rel_name in self._rels.keys()

    def get_rel(self, rel_name):
        return self._rels[rel_name]

    def get_rules(self):
        return self._rules

    def get_queries(self):
        return self._queries

    def seal(self):
        if self._sealed:
            return

        rels = list()
        for r in self._rules:
            rels.extend(r.used_rels())
        for q in self._queries:
            rels.extend(q.used_rels())
        self._rels_set = frozenset(rels)
        self._sealed = True

        for rel in self._rels_set:
            self._rels[str(rel.name())] = HornRelation(rel, env=self._env)

    def __str__(self):
        out = io.StringIO()
        for r in self._rules:
            out.write(str(r))
            out.write("\n")
        out.write("\n")
        for q in self._queries:
            out.write(str(q))
        return out.getvalue()

    def load_from_fp(self, fp, queries):
        assert fp.ctx == self._ctx
        self._fp = fp
        if len(queries) > 0:
            for r in fp.get_rules():
                rule = HornRule(r)
                self.add_rule(rule)
            for q in queries:
                rule = HornRule(z3.Implies(q, False))
                self.add_rule(rule)
        else:
            # fixedpoint object is not properly loaded, ignore it
            self._fp = None
            for a in fp.get_assertions():
                rule = HornRule(a)
                self.add_rule(rule)
        self.seal()

    def has_fixedpoint(self):
        return self._fp is not None

    def get_fixedpoint(self):
        return self._fp

    def mk_fixedpoint(self, fp=None):
        if fp is None:
            self._fp = z3.Fixedpoint(ctx=self._ctx)
            fp = self._fp

        fp_ctx = fp.ctx
        if fp_ctx == self._ctx:
            def trans(x):
                return x
        else:
            def trans(x):
                return x.translate(fp_ctx)

        for rel in self._rels_set:
            fp.register_relation(trans(rel))
        for r in self._rules:
            if r.has_formula():
                fp.add_rule(trans(r.get_formula()))
            else:
                fp.add_rule(trans(r.mk_formula()))

        return fp

    def get_ctx(self):
        return self._ctx


class FolModel(object):
    def __init__(self):
        self._fn_interps = dict()

    def add_fn(self, name, lmbd):
        self._fn_interps[name] = lmbd

    def has_interp(self, name):
        return name in self._fn_interps.keys()

    def __setitem__(self, key, val):
        self.add_fn(key, val)

    def get_fn(self, name):
        return self._fn_interps[name]

    def eval(self, term):
        fn = self.get_fn(term.decl().name())
        # lambdas seem to be broken at the moment
        # this is a work around
        body = fn.body()
        body = z3.substitute_vars(body, *reversed(term.children()))
        return body

    def __str__(self):
        return str(self._fn_interps)

    def __repr__(self):
        return repr(self._fn_interps)


def load_horn_db_from_file(fname, context=z3.main_ctx(),
                           simplify_queries=True, env=None):
    fp = z3.Fixedpoint(ctx=context)
    queries = fp.parse_file(fname)
    db = HornClauseDb(fname, ctx=context,
                      simplify_queries=simplify_queries, env=env)
    db.load_from_fp(fp, queries)
    return db


# noinspection PyProtectedMember
def main():
    db = load_horn_db_from_file(sys.argv[1], z3.main_ctx())
    print(db)
    print(db.get_rels())
    print(db._rels)
    rel = db.get_rel("main@_bb723")
    lemma_stream = io.StringIO(
        "(=> (< main@_bb723_0_n 1) (>= (+ main@_bb723_4_n main@_bb723_5_n) 0))"
    )
    lemma = rel.pysmt_parse_lemma(lemma_stream)
    print(lemma)
    print(lemma._content._asdict())
    # import json
    # json.dumps(lemma._content._asdict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
