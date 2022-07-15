# coding: utf-8
import argparse
import logging
import time
from typing import List

import z3
from z3.z3util import get_vars

from bvef_baselines import qsmt, cegar_efsmt
from exists_solver import ExistsSolver
from forall_solver import ForAllSolver

"""
Solving Exits-Forall Problem (currently focus on bit-vec?)

https://github.com/pysmt/pysmt/blob/97088bf3b0d64137c3099ef79a4e153b10ccfda7/examples/efsmt.py


Possible extensions:
- better generalizaton for esolver
- better generalizaton for fsolver
- uniform sampling for processing multiple models each round?

- use unsat core??

However, the counterexample may not be general enough to exclude a large class of invalid expressions, 
which will lead to the repetition of several loop iterations. We believe our sampling technique could 
be a good enhancement to CEGIS. By generating several diverse counterexamples, the verifier can 
provide more information to the learner so that it can make more progress on its own,
limiting the number of calls to the verifier
"""


def bv_efsmt_with_uniform_sampling(y: List, phi: z3.ExprRef, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    FIXME: inconsistent with efsmt
    """
    x = [item for item in get_vars(phi) if item not in y]

    esolver = ExistsSolver(x, z3.BoolVal(True))
    fsolver = ForAllSolver()
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("Round: ", loops)
        # TODO: need to make the fist and the subsequent iteration different???
        # TODO: in the uniform sampler, I always call the solver once before xx...
        emodels = esolver.get_models(3)

        if len(emodels) == 0:
            return z3.unsat  # esolver tells unsat
        else:
            sub_phis = []
            reverse_sub_phis = []
            for emodel in emodels:
                tau = [emodel.eval(var, True) for var in x]
                sub_phi = phi
                for i in range(len(x)):
                    sub_phi = z3.simplify(z3.substitute(sub_phi, (x[i], tau[i])))
                sub_phis.append(sub_phi)
                reverse_sub_phis.append(z3.Not(sub_phi))

            fsolver.push()  # currently, do nothing
            res_label, fmodels = fsolver.check(reverse_sub_phis)
            # print(fmodels)
            if 0 in res_label:  # there is at least one unsat Not(sub_phi)
                return z3.sat  # fsolver tells sat
            else:
                # refine using all subphi
                # TODO: the fsolver may return sigma, instead of the models
                for fmodel in fmodels:
                    sigma = [fmodel.eval(vy, True) for vy in y]
                    sub_phi = phi
                    for j in range(len(y)):
                        sub_phi = z3.simplify(z3.substitute(sub_phi, (y[j], sigma[j])))
                    # block all CEX?
                    esolver.fmls.append(sub_phi)
                    fsolver.pop()

    return z3.unknown


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    # '''
    # fmlb = And(y > 3, x == 1)
    start = time.time()
    print(cegar_efsmt([y], fmla, 100))
    print(time.time() - start)

    start = time.time()
    print(bv_efsmt_with_uniform_sampling([y], fmla, 100))
    print(time.time() - start)
    # '''

    start = time.time()
    print(qsmt([y], fmla))
    print(time.time() - start)


def test():
    w, x, y, z = z3.Ints("w x y z")
    fml = z3.And(x <= y, z <= z3.If(x >= w, x, w))

    start = time.time()
    qfml = z3.ForAll([x, y, z], fml)
    s = z3.SolverFor("UFBV")
    # s = Solver()
    s.add(qfml)
    print(s.check())  # unsat
    print(time.time() - start)

    start = time.time()
    print(cegar_efsmt([x, y, z], fml))
    print(time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='strategy', default='default', type=str,
                        help="qsmt, efsmt")
    parser.add_argument('--file', dest='file', default='none', type=str, help="file")
    parser.add_argument('--timeout', dest='timeout', default=30, type=int, help="timeout")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug("Start to solve")

    # test_efsmt()
    test()
