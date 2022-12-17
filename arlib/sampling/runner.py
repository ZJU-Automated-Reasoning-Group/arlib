import z3
from arlib.sampling.lira_sampler import LRASampler

"""
Command-line tools for sampling...
"""


class SMTSampler(object):

    def __init__(self):
        self.formula = None

    def init_from_file(self, file_name: str):
        self.formula = z3.And(z3.parse_smt2_file(file_name))

    def sample_smt_lra_models(self):
        sampler = LRASampler()
        sampler.init_from_smt(self.formula)
        sampler.sample(3)

    def sample_smt_bool_models(self):
        raise NotImplementedError

    def sample_smt_bv_models(self):
        raise NotImplementedError


class DIMACSSampler(object):

    def __init__(self):
        self.formula = None

    def init_from_file(self, file_name: str):
        self.__parse_dimacs(file_name)
        self.formula = None

    def __parse_dimacs(self, file_name: str):
        return

    def sample_dimacs_bool_models(self):
        raise NotImplementedError
