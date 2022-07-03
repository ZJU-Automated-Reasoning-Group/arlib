"""
Propositional Interpolant
"""
from typing import List
import z3

from pysat.formula import CNF
from pysat.solvers import Solver


class Z3Interpolant:

    @staticmethod
    def mk_lit(m: z3.ModelRef, x: z3.ExprRef):
        if z3.is_true(m.eval(x)):
            return x
        else:
            return z3.Not(x)

    @staticmethod
    def pogo(A: z3.Solver, B: z3.Solver, xs: List[z3.ExprRef]):
        """
        Z3-based implementation
        """
        while z3.sat == A.check():
            m = A.model()
            L = [Z3Interpolant.mk_lit(m, x) for x in xs]
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
        return list(Z3Interpolant.pogo(solver_a, solver_b, var_list))


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


def test_itp():
    a1, a2, b1, b2, x1, x2 = z3.Bools('a1 a2 b1 b2 x1 x2')
    fml_a = z3.And(a1, a2, b1)
    fml_b = z3.Or(z3.Not(a1), z3.Not(a2))
    print(list(Z3Interpolant.compute_itp(fml_a, fml_b, [a1, a2])))