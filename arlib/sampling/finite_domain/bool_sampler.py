"""
Sampler for Boolean formulas
"""

from typing import List, Dict
from arlib.sampling.utils.sampler import Sampler


class BoolSampler(Sampler):
    """For SAT formulas"""

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.conjuntion_sampler = None
        self.number_samples = 0

    def sample(self, number: int = 1) -> List[Dict[str, bool]]:
        """
        External interface for sampling
        :param number: Number of samples to generate (default: 1)
        :return: List of sampled models, each represented as a dictionary
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def sample_via_enumeration(self) -> List[Dict[str, bool]]:
        """
        Sample via enumeration of models
        :return: List of sampled models, each represented as a dictionary
        """
        raise NotImplementedError("Enumeration-based sampling not implemented")

    def sample_via_smt_enumeration(self) -> List[Dict[str, bool]]:
        """
        Call an SMT solver iteratively (block sampled models)
        :return: List of sampled models, each represented as a dictionary
        """
        raise NotImplementedError("SMT enumeration-based sampling not implemented")

    def sample_via_knowledge_compilation(self) -> List[Dict[str, bool]]:
        """
        Translate the formula to some special forms in the knowledge compilation community
        :return: List of sampled models, each represented as a dictionary
        """
        raise NotImplementedError("Knowledge compilation-based sampling not implemented")

    def sample_via_smt_random_seed(self) -> List[Dict[str, bool]]:
        """
        Call an SMT solver iteratively (no blocking, but give the solver different random seeds)
        :return: List of sampled models, each represented as a dictionary
        """
        raise NotImplementedError("SMT random seed-based sampling not implemented")
