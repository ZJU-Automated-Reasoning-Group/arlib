"""
Propositional Interpolant

Perhaps integrating the following implementations
 -  https://github.com/fslivovsky/interpolatingsolver/tree/9050db1d39213e94f9cadd036754aed69a1faa5f
    (it uses C++ and some third-party libraries)
"""
from typing import List

import z3
from pysat.formula import CNF
from pysat.solvers import Solver


class BooleanInterpolant:
    @staticmethod
    def mk_lit(m: z3.ModelRef, x: z3.ExprRef):
        if z3.is_true(m.eval(x)):
            return x
        else:
            return z3.Not(x)

    @staticmethod
    def pogo(A: z3.Solver, B: z3.Solver, xs: List[z3.ExprRef]):
        while z3.sat == A.check():
            m = A.model()
            L = [BooleanInterpolant.mk_lit(m, x) for x in xs]
            if z3.unsat == B.check(L):
                notL = z3.Not(z3.And(B.unsat_core()))
                yield notL
                A.add(notL)
            else:
                print("expecting unsat")
                break

    @staticmethod
    def compute_itp(fml_a: z3.ExprRef, fml_b: z3.ExprRef, var_list: List[z3.ExprRef]) -> List[z3.ExprRef]:
        solver_a = z3.SolverFor("QF_FD")
        solver_a.add(fml_a)
        solver_b = z3.SolverFor("QF_FD")
        solver_b.add(fml_b)
        return list(BooleanInterpolant.pogo(solver_a, solver_b, var_list))


class PySATInterpolant:

    @staticmethod
    def compute_itp(fml_a: CNF, fml_b: CNF, xs: List[int]):

        solver_a = Solver(bootstrap_with=fml_a)
        solver_b = Solver(bootstrap_with=fml_b)
        while solver_a.solve():
            m = solver_a.get_model()
            # TODO: check the value of a var in the model, and build assumption
            cube = []
            if solver_b.solver(assumptions=cube):
                core = solver_b.get_core()
                not_core = [-v for v in core]
                yield not_core
                solver_a.add(not_core)
            else:
                print("expecting unsat")
                break
