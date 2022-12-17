# coding: utf-8

from arlib.sampling.sampler import Sampler


class BitVecSampler(Sampler):

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

    def sample_via_knowledge_compilation(self):
        """
        Translate the formula to some special forms in the
        knowledge compilation community
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError