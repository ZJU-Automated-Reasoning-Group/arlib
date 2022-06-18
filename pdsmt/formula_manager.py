# coding: utf-8
"""

# TODO: currently, I only remove redundant ones in the unsat cores.
#  Actually, our goal is to build blocking clauses from the unsat cores.
#   (e.g., let "1 and 2 and 4" be a core, the blocking clause should be "-1 or -2 or -4"
#  So, another strategy is to build the blocking clauses first, and then use
#  the simplifier in bool.cnfsimplifier (which has many features)
"""

import itertools
from typing import List


# logger = logging.getLogger(__name__)


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
    cores.sort()
    return list(cores for cores, _ in itertools.groupby(cores))


class BooleanFormulaManager(object):
    """
    Track the correlations between Boolean variables and theory atoms
    """

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial cnt in SMT2 (without "assert")

        self.numeric_clauses = []  # initial cnt in numerical clauses
        self.bool_vars_name = []  # p@0, p@1, ...
        self.num2vars = {}  # 0 -> p@0, ...
        self.vars2num = {}  # p@0 -> 1


class TheoryFormulaManager(object):
    """
    TBD
    """

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""
