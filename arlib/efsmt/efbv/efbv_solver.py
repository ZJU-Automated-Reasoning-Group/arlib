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

import logging
import time
from arlib.efsmt.efbv.efbv_forall_solver import ForAllSolver
from arlib.efsmt.efbv.efbv_exists_solver import ExistsSolver
from arlib.efsmt.efbv.efbv_utils import EFBVResult
import z3
from z3.z3util import get_vars

logger = logging.getLogger(__name__)


def bv_efsmt_with_uniform_sampling(exists_vars, forall_vars, phi, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    FIXME: inconsistent with efsmt
    """
    # x = [item for item in get_vars(phi) if item not in y]

    esolver = ExistsSolver(exists_vars, z3.BoolVal(True))
    fsolver = ForAllSolver()
    fsolver.vars = forall_vars
    fsolver.phi = phi

    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        logger.debug("  Round: {}".format(loops))
        # TODO: need to make the fist and the subsequent iteration different???
        # TODO: in the uniform sampler, I always call the solver once before xx...
        e_models = esolver.get_models(5)

        if len(e_models) == 0:
            logger.debug("  Success with UNSAT")
            return EFBVResult.UNSAT  # esolver tells unsat
        else:
            sub_phis = []
            reverse_sub_phis = []
            for emodel in e_models:
                tau = [emodel.eval(var, True) for var in exists_vars]
                sub_phi = phi
                for i in range(len(exists_vars)):
                    sub_phi = z3.simplify(z3.substitute(sub_phi, (exists_vars[i], tau[i])))
                sub_phis.append(sub_phi)
                reverse_sub_phis.append(z3.Not(sub_phi))

            blocking_fmls = fsolver.get_blocking_fml(reverse_sub_phis)
            if z3.is_false(blocking_fmls):  # At least one Not(sub_phi) is UNSAT
                logger.debug("  Success with SAT")
                return EFBVResult.SAT  # fsolver tells sat
            # block all CEX?
            esolver.fmls.append(blocking_fmls)
            """
            fsolver.push()  # currently, do nothing
            f_models = fsolver.check(reverse_sub_phis)
            # print(fmodels)
            if len(f_models) == 0:  # At least one Not(sub_phi) is UNSAT
                logger.debug("  Success with SAT")
                return EFBVResult.SAT  # fsolver tells sat
            else:
                logger.debug(" Refining with {0} counter-examples".format(len(f_models)))
                # refine using all subphi
                # TODO: the fsolver may return sigma, instead of the models
                for fmodel in f_models:
                    sigma = [fmodel.eval(vy, True) for vy in forall_vars]
                    sub_phi = phi
                    for j in range(len(forall_vars)):
                        sub_phi = z3.simplify(z3.substitute(sub_phi, (forall_vars[j], sigma[j])))
                    # block all CEX?
                    esolver.fmls.append(sub_phi)
                    fsolver.pop()
            """
    return EFBVResult.UNKNOWN


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    # '''
    start = time.time()
    print(bv_efsmt_with_uniform_sampling([x], [y], fmla, 100))
    print(time.time() - start)


if __name__ == '__main__':
    test_efsmt()
