# coding: utf-8
import logging
import itertools
from typing import List

logger = logging.getLogger(__name__)


def merge_unsat_cores(cores: List):
    """
    Remove subsumed and redundant cores
    :param cores: a set of unsat cores
    :return:

    Consider the following cores
     [[-1, 2], [4], [5, 6, 2], [-1, 2, 3] [4]]

    - [4] and [4] are redundant
    - [-1, 2] subsumes [-1, 2, 3]?

    """
    # TODO: currently, I only remove redundant ones
    cores.sort()
    return list(cores for cores, _ in itertools.groupby(cores))


class BooleanFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial cnt in SMT2 (without "assert")

        self.numeric_clauses = []  # initial cnt in numerical clauses
        self.bool_vars_name = []  # p@0, p@1, ...
        self.num2vars = {}  # 0 -> p@0, ...
        self.vars2num = {}  # p@0 -> 1


class TheoryFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""


def test():
    import random
    cores = []
    for _ in range(1000):
        core_len = random.randint(2, 8)
        cores.append([random.randint(-10, 10) for _ in range(core_len)])
    # print(cores)
    print(len(cores))
    new_cores = merge_unsat_cores(cores)
    print(len(new_cores))

test()