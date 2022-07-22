"""Sequential version"""

import logging
from typing import List

from pysat.formula import CNF
from pysat.solvers import Solver
import z3
from z3.z3util import get_vars


logger = logging.getLogger(__name__)

sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41', 'lingeling',
               'maplechrono', 'maplecm', 'maplesat', 'minicard', 'mergesat3',   'minisat22', 'minisat-gh']


class PropSolver(object):
    def __init__(self, solver="cadical"):
        self.solver_name = solver
        self._solver = Solver(name=solver)
        self._clauses = []
        self.parallel_sampling = False  # parallel sampling of satisfying assignments
        # reduce the size of each sampled model
        self.reduce_samples = True
        self.parallel_reduce = False  # parallel reduce

    def check_sat(self):
        return self._solver.solve()

    def add_clause(self, clause: List[int]):
        self._solver.add_clause(clause)
        self._clauses.append(clause)

    def add_clauses(self, clauses: List[List[int]]):
        for cls in clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def add_cnf(self, cnf: CNF):
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def get_model(self):
        return self._solver.get_model()


def simple_cegar_efsmt_bv(y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    """ Solve exists x. forall y. phi(x, y) with simple CEGAR
    """
    x = [item for item in get_vars(phi) if item not in y]
    # set_param("verbose", 15)
    qf_logic = "QF_BV"  # or QF_UFBV
    esolver = z3.SolverFor(qf_logic)
    fsolver = z3.SolverFor(qf_logic)
    esolver.add(z3.BoolVal(True))
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("round: ", loops)
        eres = esolver.check()
        if eres == z3.unsat:
            return z3.unsat
        else:
            emodel = esolver.model()
            mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
            sub_phi = z3.simplify(z3.substitute(phi, mappings))
            fsolver.push()
            fsolver.add(z3.Not(sub_phi))
            if fsolver.check() == z3.sat:
                fmodel = fsolver.model()
                y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                esolver.add(sub_phi)
                fsolver.pop()
            else:
                return z3.sat
    return z3.unknown


def test_prop():
    cnf = CNF(from_clauses=[[1, 3], [-1, 2, -4], [2, 4]])
    # solver_name = random.choice(sat_solvers)
    sol = PropSolver()
    sol.add_cnf(cnf)
    print(sol.check_sat())


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    # '''
    # fmlb = And(y > 3, x == 1)
    res = simple_cegar_efsmt_bv([y], fmla)
    print(res)


if __name__ == "__main__":
    test_prop()
    test_efsmt()

