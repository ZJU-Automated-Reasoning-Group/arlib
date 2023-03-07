# coding: utf-8
"""
Flattening-based QF_BV solver
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.utils import SolverResult

logger = logging.getLogger(__name__)


class QFFPSolver:

    def __init__(self):
        self.fml = None
        # self.vars = []
        self.verbose = 0

    def from_smt_file(self, filepath: str):
        fml_vec = z3.parse_smt2_file(filepath)
        self.fml = z3.And(fml_vec)

    def from_smt_string(self, smt: str):
        fml_vec = z3.parse_smt2_string(smt)
        self.fml = z3.And(fml_vec)

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula

    def check_sat(self):
        """Check satisfiability of an QF_FP formula"""
        logger.debug("Start translating to CNF...")
        qffp_preamble = z3.AndThen(z3.With('simplify', arith_lhs=False, elim_and=True),
                                   'propagate-values',
                                   'fpa2bv',
                                   'propagate-values',
                                   z3.With('simplify', arith_lhs=False, elim_and=True),
                                   'ackermannize_bv',
                                   'bit-blast',
                                   # If we do not add the following pass, the tseitin-cnf tactic may report an errror
                                   z3.With('simplify', arith_lhs=False, elim_and=True)
                                   )

        after_simp = qffp_preamble(self.fml).as_expr()
        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        elif z3.is_true(after_simp):
            return SolverResult.SAT

        g_probe = z3.Goal()
        g_probe.add(after_simp)
        is_bool = z3.Probe('is-propositional')
        if is_bool(g_probe) == 1.0:
            to_cnf = z3.AndThen(z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False), 'tseitin-cnf')
            qfbv_tactic = z3.With(to_cnf, elim_and=True, push_ite_bv=True, blast_distinct=True)
            blasted = qfbv_tactic(after_simp).as_expr()
            g_to_dimacs = z3.Goal()
            g_to_dimacs.add(blasted)
            pos = CNF(from_string=g_to_dimacs.dimacs())
            aux = Solver(name="minisat22", bootstrap_with=pos)
            if aux.solve():
                return SolverResult.SAT
            return SolverResult.UNSAT
        else:
            sol = z3.Tactic('smt').solver()
            sol.add(after_simp)
            res = sol.check()
            if res == z3.sat:
                return SolverResult.SAT
            elif res == z3.unsat:
                return SolverResult.UNSAT
            else:
                return SolverResult.UNKNOWN


def demo_qffp():
    z3.set_param("verbose", 15)
    fml_str = """
        (declare-fun X () (_ FloatingPoint 2 6))
    (declare-fun Y () (_ FloatingPoint 2 6))
    (declare-fun Z () (_ FloatingPoint 2 6))
    (assert (and (= X (fp.add RTZ Y Z)) (= X (fp.div RTZ Y Z)) (= X (fp.roundToIntegral RTZ Y)) (not (= Y Z))))
    (check-sat)
        """

    sol = QFFPSolver()
    sol.from_smt_string(fml_str)
    print(sol.check_sat())


if __name__ == "__main__":
    demo_qffp()
