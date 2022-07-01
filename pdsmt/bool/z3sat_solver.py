# coding: utf-8
from typing import List

import z3


class Z3SATSolver:

    def __init__(self, logic="QF_FD"):
        self.int2z3var = {}  # for initializing from int clauses
        self.solver = z3.SolverFor(logic)
        # self.solver = z3.SimpleSolver()

    def add_clauses(self, clauses: List[List[int]]):
        """
        Initialize self.solver with a list of clauses
        """
        # z3_clauses = []
        for clause in clauses:
            conds = []
            for t in clause:
                if t == 0: break
                a = abs(t)
                if a in self.int2z3var:
                    b = self.int2z3var[a]
                else:
                    b = z3.Bool("b!{}".format(a))
                    self.int2z3var[a] = b
                b = z3.Not(b) if t < 0 else b
                conds.append(b)
            self.solver.add(z3.Or(*conds))

    def get_z3var(self, vid: int):
        """
        Given an integer (labeling a Boolean var.), return its corresponding Z3 Boolean var
        """
        if vid in self.int2z3var:
            return self.int2z3var[vid]
        raise Exception(str(vid) + " not in the var list!")

    def get_unsat_core(self, assumptions: List[z3.BoolRef]):
        if self.solver.check(assumptions) == z3.unsat:
            return self.solver.unsat_core()
        return []

    def check_sat(self):
        return self.solver.check()
