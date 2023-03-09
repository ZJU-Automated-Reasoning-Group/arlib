# coding: utf-8
"""
Flattening-based QF_FP solver
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.utils import SolverResult

logger = logging.getLogger(__name__)


class QFFPSolver:
    """
    This class is used to check the satisfiability of QF_FP formulas. It uses various tactics from Z3
    to translate the formula to CNF and then use PySAT to solve it.
    """

    def __init__(self):
        self.fml = None
        # self.vars = []
        self.verbose = 0

    def from_smt_file(self, filepath: str):
        fml_vec = z3.parse_smt2_file(filepath)
        self.fml = z3.And(fml_vec)

    def from_smt_string(self, smt: str):
        """
        Parse an SMT string and set the class member fml to the corresponding formula contained in the string.
        :param smt: An SMT-LIB2 string
        """
        fml_vec = z3.parse_smt2_string(smt)
        self.fml = z3.And(fml_vec)

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula

    def check_sat(self):
        # z3.set_param("verbose", 15)
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

        try:
            after_simp = qffp_preamble(self.fml).as_expr()
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
                logger.debug("Running pysat...")
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

        except Exception as ex:
            print("ERROR")
            # exit(0)
            print(ex)
            s = z3.Solver()
            s.add(self.fml)
            print(s.to_smt2())
            return SolverResult.UNKNOWN


def demo_qffp():
    z3.set_param("verbose", 15)
    fml_str = """
(set-logic QF_FP)
 (declare-fun fpv1 () Float32)
 (declare-fun fpv2 () Float32)
 (declare-fun fpv3 () Float32)
 (declare-fun v0 () Bool)
 (declare-fun v1 () Bool)
 (declare-fun v2 () Bool)
 (declare-fun fpv6 () Float32)
 (declare-fun fpv7 () Float32)
 (declare-fun v3 () Bool)
 (declare-fun v4 () Bool)
 (declare-fun fpv8 () Float32)
 (declare-fun v5 () Bool)
 (declare-fun v6 () Bool)
 (assert (or (or v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24))) (fp.isPositive fpv7)) false))
 (assert (or (or v5 (fp.isPositive fpv7) v3) false))
 (assert (or (or v5 (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) false))
 (assert (or (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1))) false))
 (assert (or (or v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24))) v3) false))
 (assert (or (or v5 (fp.isPositive fpv7) v3) false))
 (assert (or (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7))) false))
 (assert (or (or v3 v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)))) false))
 (assert (or (or (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24))) v5 v5) false))
 (assert (or (or v5 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v5) false))
 (assert (or (or (fp.isPositive fpv7) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1))) false))
 (assert (or (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1))) false))
 (assert (or (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) false))
 (assert (or (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) false))
 (assert (or v3 v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)))))
 (assert (or (fp.isPositive fpv7) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1))))
 (assert (and (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)))))
 (assert (and (and (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)))) (or v5 (fp.isPositive fpv7) v3) (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) (or v3 v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)))) (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)))))
 (assert (=> (or v3 (not (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24))) (fp.isPositive fpv7)) (and (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7))))))
 (assert (or (or v5 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v5) (and (or v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1))) (or (xor v1 (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1) (xor (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) (fp.geq fpv8 fpv2 ((_ to_fp 8 24) RNA 63910191.0) (_ -oo 8 24)) (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v4 (xor (or v1 v2 v1) (or v1 v2 v1) (xor v0 (fp.isNormal (_ +oo 8 24)) (fp.isNormal (_ +oo 8 24)) v1 (fp.isNormal (_ +oo 8 24)) (or v1 v2 v1) (or v1 v2 v1)) (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) (or v1 v2 v1) (fp.isPositive fpv7) (fp.isPositive fpv7)) v1)) (fp.isPositive fpv7) v5) (or (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7)) v3 (or (fp.eq (_ +zero 8 24) fpv3 (_ NaN 8 24) (_ -oo 8 24) (_ +oo 8 24)) v2 v2 (fp.isPositive fpv7))))))
 (check-sat)
        """

    sol = QFFPSolver()
    sol.from_smt_string(fml_str)
    print(sol.check_sat())


if __name__ == "__main__":
    demo_qffp()