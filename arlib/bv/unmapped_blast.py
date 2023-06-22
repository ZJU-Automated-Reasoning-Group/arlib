import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from arlib.utils.typing import SolverResult

qfbv_preamble = z3.AndThen(z3.With('simplify', flat_and_or=False),
                           z3.With('propagate-values', flat_and_or=False),
                           z3.Tactic('elim-uncnstr'),
                           z3.With('solve-eqs', solve_eqs_max_occs=2),
                           z3.Tactic('reduce-bv-size'),
                           z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                                   local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
                           # Z3 can solve a couple of extra benchmarks by using hoist_mul but the timeout in SMT-COMP is too small.
                           # Moreover, it impacted negatively some easy benchmarks. We should decide later, if we keep it or not.
                           # With('simplify', hoist_mul=False, som=False, flat_and_or=False),
                           z3.Tactic('max-bv-sharing'),
                           z3.Tactic('ackermannize_bv'),
                           z3.Tactic('bit-blast'),
                           # z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
                           # With('solve-eqs', local_ctx=True, flat=False, flat_and_or=False),
                           z3.Tactic('tseitin-cnf')
                           )

qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)


def qfbv_to_sat(fml):
    after_simp = qfbv_tactic(fml).as_expr()
    if z3.is_false(after_simp):
        return SolverResult.UNSAT
    elif z3.is_true(after_simp):
        return SolverResult.SAT
    g = z3.Goal()
    g.add(after_simp)
    pos = CNF(from_string=g.dimacs())
    aux = Solver(name="minisat22", bootstrap_with=pos)
    print("solving via pysat")
    if aux.solve():
        return SolverResult.SAT
    return SolverResult.UNSAT


def demo():
    z3.set_param("verbose", 15)
    x, y = z3.BitVecs("x y", 6)
    fml = z3.And(x + y == 8, x - y == 2)
    print(qfbv_to_sat(fml))


demo()
