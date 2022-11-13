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
from arlib.utils.exceptions import ExitsSolverSuccess, ExitsSolverUnknown, ForAllSolverSuccess, ForAllSolverUnknown

import z3
from z3.z3util import get_vars
from z3 import *

logger = logging.getLogger(__name__)


def bv_efsmt_with_uniform_sampling(exists_vars, forall_vars, phi, maxloops=None):
    """
    Solves exists x. forall y. phi(x, y)
    FIXME: inconsistent with efsmt
    """
    # x = [item for item in get_vars(phi) if item not in y]

    esolver = ExistsSolver(exists_vars, z3.BoolVal(True))
    fsolver = ForAllSolver(exists_vars[0].ctx)
    # fsolver.vars = forall_vars
    # fsolver.phi = phi

    loops = 0
    result = EFBVResult.UNKNOWN
    try:
        while maxloops is None or loops <= maxloops:
            logger.debug("  Round: {}".format(loops))
            loops += 1
            # TODO: need to make the fist and the subsequent iteration different???
            # TODO: in the uniform sampler, I always call the solver once before xx...
            e_models = esolver.get_models(5)

            if len(e_models) == 0:
                logger.debug("  Success with UNSAT")
                result = EFBVResult.UNSAT  # esolver tells unsat
                break
            else:
                # sub_phis = []
                reverse_sub_phis = []
                # print("e models: ", e_models)
                for emodel in e_models:
                    x_mappings = [(x, emodel.eval(x, model_completion=True)) for x in exists_vars]
                    sub_phi = z3.simplify(z3.substitute(phi, x_mappings))
                    # sub_phis.append(sub_phi)
                    reverse_sub_phis.append(z3.Not(sub_phi))

                fmodels = fsolver.check(reverse_sub_phis)
                if len(fmodels) == 0:
                    logger.debug("  Success with SAT")
                    result = EFBVResult.SAT  # fsolver tells sat
                    break
                for fmodel in fmodels:
                    # sigma = [model.eval(vy, True) for vy in self.forall_vars]
                    y_mappings = [(y, fmodel.eval(y, model_completion=True)) for y in forall_vars]
                    sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                    # block all CEX?
                    # print("blocking fml: ", sub_phi)
                    if z3.is_false(sub_phi):
                        logger.debug("  Success with UNSAT")
                        # using break is not fine
                        raise ExitsSolverSuccess()
                    esolver.fmls.append(sub_phi)

    except ForAllSolverSuccess as ex:
        # print(ex)
        logger.debug("  Forall solver SAT")
        result = EFBVResult.SAT
    except ForAllSolverUnknown as ex:
        logger.debug("  Forall solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except ExitsSolverSuccess as ex:
        logger.debug("  Exists solver UNSAT")
        result = EFBVResult.UNSAT
    except ExitsSolverUnknown as ex:
        logger.debug("  Exists solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except Exception as ex:
        print("XX")

    return result


def test_efsmt():
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    # '''
    start = time.time()
    print(bv_efsmt_with_uniform_sampling([x], [y], fmla, 100))
    print(time.time() - start)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_efsmt()
