# coding: utf-8
"""
This module provides a MaxSATSolver class that wraps different MaxSAT engines and implements
methods for solving weighted and unweighted MaxSAT problems. It also includes an implementation
of Nadel's algorithm for OMT(BV) "Bit-Vector Optimization (TACAS'16)".
"""

import copy

from pysat.formula import WCNF
from pysat.solvers import Solver

from .fm import FM  # is the FM correct???
from .rc2 import RC2


class MaxSATSolver:
    """
    Wrapper of the engines in maxsat
    """

    def __init__(self, formula: WCNF):
        """
        :param formula: input MaxSAT formula
        """
        self.maxsat_engine = "FM"
        self.wcnf = formula
        self.hard = copy.deepcopy(formula.hard)
        self.soft = copy.deepcopy(formula.soft)
        self.weight = formula.wght[:]

        self.sat_engine_name = "m22"
        # g3, g4, lgl, mcb, mcm, mpl, m22, mc, mgh, z3

    def set_maxsat_engine(self, name: str):
        self.maxsat_engine = name

    def get_maxsat_engine(self):
        """Get MaxSAT engine"""
        return self.maxsat_engine

    def solve_wcnf(self):
        """TODO: support Popen-based approach for calling bin solvers (e.g., open-wbo)"""
        if self.maxsat_engine == "FM":
            fm = FM(self.wcnf, verbose=0)
            fm.compute()
            # print("cost, ", fm.cost)
            return fm.cost
        elif self.maxsat_engine == "RC2":
            rc2 = RC2(self.wcnf)
            rc2.compute()
            return rc2.cost
        else:
            fm = FM(self.wcnf, verbose=0)
            fm.compute()
            return fm.cost

    def tacas16_binary_search(self):
        """
        Implement Nadel's algorithm for OMT(BV) "Bit-Vector Optimization (TACAS'16)"

        Key idea: OMT on unsigned BV can be seen as lexicographic optimization over the bits in the
        bitwise representation of the objective, ordered from the most-significant bit (MSB)
        to the least-significant bit (LSB).

        Notice that, in this domain, this corresponds to a binary search over the space of the values of the objective

        NOTE: we assume that each element in self.soft is a unary clause, i.e., self.soft is [[l1], [l2], ...]
        """
        bits = []
        for i in reversed(range(len(self.soft))):
            bits.append(self.soft[i][0])

        assumption_lits = []
        try:
            assert self.sat_engine_name != "z3"
            """Use a SAT solver supported by pySAT"""
            sat_oracle = Solver(name=self.sat_engine_name, bootstrap_with=self.hard, use_timer=True)
            # For each bit in bits, decide if it can be true
            for b in bits:
                assumption_lits.append(b)
                if not sat_oracle.solve(assumptions=assumption_lits):
                    # if b cannot be positive, then set it to be negative?
                    # after this round, we will try the remaining bits
                    assumption_lits.pop()
                    assumption_lits.append(-b)
        except Exception as ex:
            print(ex)
        # print("final assumptions: ", assumption_lits)
        return assumption_lits

def obv_bs(clauses,n):
    result = []

    s = Solver(bootstrap_with=clauses)

    if s.solve():
        m = s.get_model()
        #print(m)
    else:
        print('UNSAT')
        return result
    l = len(m)
    for i in range(l):
        if m[i] > 0:
            result.append(i+1)
        else:
            result.append(i + 1)
            if s.solve(assumptions=result):
                m = s.get_model()
            else:
                result.pop()
                result.append(-i-1)
    if l < n:
        while l < n:
            result.append(l+1)
            l = l + 1
    #print(result)
    return result

obv_bs([[1,2],[-3,-4],[-5,-7]],8)
