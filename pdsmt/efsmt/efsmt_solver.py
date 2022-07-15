# coding: utf-8
"""
A uniformed interface for solving Exists-ForAll problems
"""
import logging
import time
from typing import List

import z3

from ..utils.smtlib_solver import SMTLIBSolver
from .simple_cegar import cegar_efsmt

logger = logging.getLogger(__name__)


class EFSMTBVSolver:
    """Solving exists forall problem"""

    def __init__(self, logic: str):
        self.tactic = None
        self.logic = logic

    def solve_with_qbf(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """
        Translate EFSMT(BV) to QBF (preserve the quantifiers)
        :param y: variables to be universal quantified
        :param phi: a quantifier-free formula
        :return: a QBF formula?
        """
        assert self.logic == "BV" or self.logic == "UFBV"
        raise NotImplementedError

    def solve_with_sat(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """
        Translate EFSMT(BV) to SAT and use an SAT solver to solve
          1. First, EFSMT(BV) to QBF
          2. Second, QBF to SAT (maybe use expansion-based QE?)
        :param y: variables to be universal quantified
        :param phi: a quantifier-free formula
        :return: a QBF formula?
        """
        assert self.logic == "BV" or self.logic == "UFBV"
        raise NotImplementedError

    def solve_with_simple_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """
        Solve with a CEGAR-style algorithm, which consists of a "forall solver" and an "exists solver"
        :return:
        """
        """This can be slow (perhaps not a good idea for NRA) Maybe good for LRA or BV?"""
        print("User-defined EFSMT starting!!!")
        z3_res, model = cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_qsmt(self, y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
        """TODO: should we use smtlib_solver to support more engines? (instead of just z3)
        """
        try:
            # set_param("verbose", 15)
            qfml = z3.ForAll(y, phi)
            # print(qfml)
            s = z3.SolverFor("UFBV")  # can be very fast
            # s = Solver()
            s.add(qfml)
            return s.check()
        except Exception as ex:
            print(ex)

    def solve_with_bin_smt(self, y, phi: z3.ExprRef):
        """
        Build a quantified formula and send it to a third-party SMT solver
        that can handle quantified formulas
        :param y: the universal quantified variables
        :param phi: the quantifier-free part of the VC
        :return: TBD
        """
        smt2string = "(set-logic {})\n".format(self.logic)
        sol = z3.Solver()
        sol.add(z3.ForAll(y, phi))
        smt2string += sol.to_smt2()

        bin_cmd = f"/Users/prism/Work/cvc5/build/bin/cvc5 -q --produce-models"
        bin_solver = SMTLIBSolver(bin_cmd)
        start = time.time()
        res = bin_solver.check_sat_from_scratch(smt2string)
        if res == "sat":
            # print(bin_solver.get_expr_values(["p1", "p0", "p2"]))
            print("External solver success time: ", time.time() - start)
            # TODO: get the model to build the invariant
        elif res == "unsat":
            print("External solver fails time: ", time.time() - start)
        else:
            print("Seems timeout or error in the external solver")
            print(res)
        bin_solver.stop()
