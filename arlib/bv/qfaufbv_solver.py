# coding: utf-8
"""
Flattening-based QF_AUFBV solver
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.utils import SolverResult

logger = logging.getLogger(__name__)


class QFAUFBVSolver:

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

        qfaufbv_preamble = z3.AndThen('simplify',
                                      'propagate-values',
                                      z3.With('solve-eqs'),
                                      'elim-uncnstr',
                                      'reduce-bv-size',
                                      z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False,
                                              local_ctx=True, local_ctx_limit=10000000),
                                      # 'bvarray2uf',  # this tactic is dangerous (it only handles specific arrays)
                                      'max-bv-sharing',
                                      'ackermannize_bv',
                                      z3.If(z3.Probe('is-qfbv'),
                                            z3.AndThen('bit-blast',
                                                       z3.With('simplify', arith_lhs=False, elim_and=True)),
                                            'simplify'),
                                      )

        qfaufbv = z3.With(qfaufbv_preamble, elim_and=True, sort_store=True)

        after_simp = qfaufbv(self.fml).as_expr()
        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        elif z3.is_true(after_simp):
            return SolverResult.SAT

        g_probe = z3.Goal()
        g_probe.add(after_simp)
        is_bool = z3.Probe('is-propositional')
        if is_bool(g_probe) == 1.0:
            to_cnf = z3.AndThen('simplify', 'tseitin-cnf')
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


def demo_qfaufbv():
    z3.set_param("verbose", 15)
    fml_str = """
       (set-info :smt-lib-version 2.6)
(set-logic QF_ABV)
(declare-fun size_Q_0 () (_ BitVec 32))
(declare-fun size_Q_1 () (_ BitVec 32))
(assert (= size_Q_1 (_ bv2 32)))
(declare-fun BubbleSort_Q_0_Q_j_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_j_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_j_Q_1 (_ bv0 32)))
(declare-fun BubbleSort_Q_0_Q_i_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_i_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_i_Q_1 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (_ bv0 32) BubbleSort_Q_0_Q_i_Q_0)))
(declare-fun array_Q_0 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun BubbleSort_Q_0_Q_temp_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_1 () (_ BitVec 32))
(assert (let ((?v_0 (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1))) (= BubbleSort_Q_0_Q_temp_Q_1 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult ?v_0 (select array_Q_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32))))) ?v_0 BubbleSort_Q_0_Q_temp_Q_0))))
(declare-fun array_Q_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert (let ((?v_0 (select array_Q_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32))))) (= array_Q_1 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) ?v_0)) (store array_Q_0 BubbleSort_Q_0_Q_i_Q_1 ?v_0) array_Q_0))))
(declare-fun array_Q_2 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert (let ((?v_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)))) (= array_Q_2 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) (select array_Q_0 ?v_0))) (store array_Q_1 ?v_0 BubbleSort_Q_0_Q_temp_Q_1) array_Q_1))))
(declare-fun BubbleSort_Q_0_Q_temp_Q_i_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_i_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_temp_Q_i_Q_1 (ite (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) BubbleSort_Q_0_Q_i_Q_1 BubbleSort_Q_0_Q_temp_Q_i_Q_0)))
(declare-fun BubbleSort_Q_0_Q_i_Q_2 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_i_Q_2 (ite (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)) BubbleSort_Q_0_Q_i_Q_1)))
(declare-fun BubbleSort_Q_0_Q_temp_Q_j_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_j_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_temp_Q_j_Q_1 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) BubbleSort_Q_0_Q_j_Q_1 BubbleSort_Q_0_Q_temp_Q_j_Q_0)))
(declare-fun BubbleSort_Q_0_Q_j_Q_2 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_j_Q_2 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvadd BubbleSort_Q_0_Q_j_Q_1 (_ bv1 32)) BubbleSort_Q_0_Q_j_Q_1)))
(assert (let ((?v_9 (bvsub (_ bv2 32) (_ bv1 32)))) (let ((?v_8 (bvult BubbleSort_Q_0_Q_j_Q_1 ?v_9)) (?v_7 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (let ((?v_0 (and ?v_8 (bvult BubbleSort_Q_0_Q_i_Q_1 ?v_7))) (?v_1 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)))) (let ((?v_4 (and ?v_0 (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) (select array_Q_0 ?v_1)))) (?v_2 (and (bvule (_ bv0 32) BubbleSort_Q_0_Q_i_Q_1) (bvult BubbleSort_Q_0_Q_i_Q_1 (_ bv2 32)))) (?v_5 (and (bvule (_ bv0 32) ?v_1) (bvult ?v_1 (_ bv2 32))))) (let ((?v_3 (=> ?v_4 ?v_2)) (?v_6 (=> ?v_4 ?v_5))) (not (and (and (and (and (and (and (and (and (and (and (=> ?v_0 ?v_2) (=> ?v_0 ?v_5)) ?v_3) ?v_3) ?v_6) ?v_6) (=> ?v_0 (not (bvult BubbleSort_Q_0_Q_i_Q_2 ?v_7)))) (=> ?v_8 (not (bvult BubbleSort_Q_0_Q_j_Q_2 ?v_9)))) (and (bvule (_ bv0 32) (_ bv0 32)) (bvult (_ bv0 32) (_ bv2 32)))) (and (bvule (_ bv0 32) (_ bv1 32)) (bvult (_ bv1 32) (_ bv2 32)))) (bvule (select array_Q_2 (_ bv0 32)) (select array_Q_2 (_ bv1 32)))))))))))
(check-sat)
        """
    sol = QFAUFBVSolver()
    sol.from_smt_string(fml_str)
    print(sol.check_sat())


if __name__ == "__main__":
    demo_qfaufbv()