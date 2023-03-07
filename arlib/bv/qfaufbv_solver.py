import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from arlib.utils.types import SolverResult

is_qfbv = z3.Probe('is-qfbv')

qfaufbv_preamble = z3.AndThen(z3.With('simplify'),
                              z3.With('propagate-values'),
                              z3.With('solve-eqs', solve_eqs_max_occs=2),
                              z3.Tactic('elim-uncnstr'),
                              z3.Tactic('reduce-bv-size'),
                              z3.With('solve-eqs', solve_eqs_max_occs=2),
                              z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                                      local_ctx_limit=10000000, flat=True, hoist_mul=False),
                              z3.Tactic('bvarray2uf'),
                              z3.Tactic('ackermannize_bv'),
                              z3.Tactic('max-bv-sharing'),
                              # FIXME: after the above step, the formula may not belong
                              #  to QF_BF, and we should use a Probe to decide this.
                              #  If it is not QF_BV, we may have to use the smt tactic
                              #  Otherwise, we can use the following procedures..
                              z3.If(is_qfbv, z3.AndThen('bit-blast', 'sat'), 'smt'),
                              # z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
                              # With('solve-eqs', local_ctx=True, flat=False, flat_and_or=False),
                              # z3.Tactic('tseitin-cnf')
                              )

qfaufbv_tactic = z3.With(qfaufbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True, sort_store=True)


def qfaufbv_to_sat(fml):
    after_simp = qfaufbv_tactic(fml).as_expr()
    if z3.is_false(after_simp):
        return SolverResult.UNSAT
    elif z3.is_true(after_simp):
        return SolverResult.SAT
    g = z3.Goal()
    g.add(after_simp)
    pos = CNF(from_string=g.dimacs())
    aux = Solver(name="minisat22", bootstrap_with=pos)
    if aux.solve():
        return SolverResult.SAT
    return SolverResult.UNSAT


def demo_aufbf():
    z3.set_param("verbose", 15)
    fml_str = """
    (set-logic QF_AUFBV)
(set-info :status sat)
(declare-fun bv_22-0 () (_ BitVec 1))
(declare-fun arr-8324605531633220487_-1461211092162269148-0 () (Array (_ BitVec 1) Bool))
(assert (select arr-8324605531633220487_-1461211092162269148-0 (bvlshr bv_22-0 bv_22-0)))
(check-sat)
    """

    fml = z3.And(z3.parse_smt2_string(fml_str))
    print(qfaufbv_to_sat(fml))


demo_aufbf()
