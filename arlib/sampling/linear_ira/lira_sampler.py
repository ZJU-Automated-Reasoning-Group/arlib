"""
For linear integer and real formulas
"""

import z3
from arlib.sampling.utils.sampler import Sampler
from arlib.sampling.linear_ira.dikin_walk import ConunctiveLRASampler
from arlib.utils.z3_solver_utils import to_dnf


class LRASampler(Sampler):

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.number_samples = 0
        self.formula = None

    def sample(self, number=1):
        """
        External interface
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def init_from_smt(self, expr: z3.ExprRef):
        self.formula = expr

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError

    def sample_via_dnf_and_dikin_walk(self):
        """
        1. Convert the formula to DNF
        2. For each conjunct in the DNF, call dikin walk?
        """
        dnf_fml = to_dnf(self.formula)
        clra = ConunctiveLRASampler()
        # for fml in dnf_fml:
        #    clra.sample()


class LIASampler(Sampler):

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.conjuntion_sampler = None
        self.number_samples = 0

    def sample(self, number=1):
        """
        External interface
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError
