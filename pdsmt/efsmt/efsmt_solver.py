# coding: utf-8
"""
A uniformed interface for solving Exists-ForAll problems
"""
import logging
import time
from enum import Enum
from typing import List

import z3

from .simple_cegar import cegar_efsmt
from ..global_params import z3_exec, cvc5_exec
from ..utils.smtlib_solver import SMTLIBSolver

logger = logging.getLogger(__name__)


def solve_with_bin_smt(y, phi: z3.ExprRef, logic: str, solver_name: str):
    """Call bin SMT solvers to solve exists forall"""
    smt2string = "(set-logic {})\n".format(logic)
    sol = z3.Solver()
    sol.add(z3.ForAll(y, phi))
    smt2string += sol.to_smt2()

    # TODO: build/download bin solvers in the project
    # bin_cmd = ""
    if solver_name == "z3":
        bin_cmd = z3_exec
    elif solver_name == "cvc5":
        bin_cmd = cvc5_exec + " -q --produce-models"
    else:
        bin_cmd = z3_exec

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
    return res


class EFSMTStrategy(Enum):
    Z3 = 0,  # via bin solver
    CVC5 = 1,  # via bin solver
    Boolector = 2,  # via bin solver
    Yices2 = 3,  # via bin solver
    SIMPLE_CEGAR = 4,
    QBF = 5,
    SAT = 6,
    PARALLEL_CEGAR = 7


class EFSMTSolver:
    """Solving exists forall problem"""

    def __init__(self, logic: str):
        self.tactic = EFSMTStrategy.SIMPLE_CEGAR
        self.logic = logic
        
    def set_tactic(self, name: str):
        raise NotImplementedError

    def solve(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """Translate EFSMT(BV) to QBF (preserve the quantifiers)
        :param y: variables to be universal quantified
        :param phi: a quantifier-free formula
        """
        if self.tactic == EFSMTStrategy.Z3 or self.tactic == EFSMTStrategy.CVC5 \
                or self.tactic == EFSMTStrategy.Boolector or self.tactic == EFSMTStrategy.Yices2:
            return self.solve_with_qsmt(y, phi)
        elif self.tactic == EFSMTStrategy.SIMPLE_CEGAR:
            return self.solve_with_simple_cegar(y, phi)
        elif self.tactic == EFSMTStrategy.PARALLEL_CEGAR:
            return self.solve_with_parallel_cegar(y, phi)
        else:
            return self.internal_solve(y, phi)

    def internal_solve(self, y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
        """Call Z3's Python API"""
        qfml = z3.ForAll(y, phi)
        s = z3.SolverFor(self.logic)  # can be very fast
        s.add(qfml)
        return s.check()

    def solve_with_simple_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """Solve with a CEGAR-style algorithm, which consists of a "forall solver" and an "exists solver"
        """
        """This can be slow (perhaps not a good idea for NRA) Maybe good for LRA or BV?"""
        print("Simple, sequential, CEGAR-style EFSMT!")
        z3_res, model = cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_parallel_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """This should be the focus"""
        print("Parallel EFSMT starting!!!")
        z3_res, model = cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_qsmt(self, y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
        """Solve with bin solvers"""
        if self.tactic == EFSMTStrategy.Z3:
            return solve_with_bin_smt(y, phi, self.logic, "z3")
        elif self.tactic == EFSMTStrategy.CVC5:
            return solve_with_bin_smt(y, phi, self.logic, "cvc5")
        elif self.tactic == EFSMTStrategy.Boolector:
            return solve_with_bin_smt(y, phi, self.logic, "boolector2")
        elif self.tactic == EFSMTStrategy.Yices2:
            return solve_with_bin_smt(y, phi, self.logic, "yices2")
        else:
            return self.internal_solve(y, phi)  # for special cases

    def solve_with_qbf(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        assert self.logic == "BV" or self.logic == "UFBV"
        raise NotImplementedError

    def solve_with_sat(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        raise NotImplementedError
