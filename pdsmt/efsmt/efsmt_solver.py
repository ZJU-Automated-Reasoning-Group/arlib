# coding: utf-8
"""
The uniformed interface for solving Exists-ForAll problems
"""
import logging
from enum import Enum
from typing import List

import z3

from .efsmt_utils import simple_cegar_efsmt, solve_with_bin_smt


logger = logging.getLogger(__name__)


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
        z3_res, model = simple_cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_parallel_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """This should be the focus"""
        print("Parallel EFSMT starting!!!")
        z3_res, model = simple_cegar_efsmt(self.logic, y, phi)
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
