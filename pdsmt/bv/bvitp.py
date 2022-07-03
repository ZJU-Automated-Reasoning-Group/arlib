# coding: utf-8
"""
1. Translate a bit-vector optimization problem to a weighted MaxSAT problem,
2. Call a third-path MaxSAT solver

TODO:
- Need to track the relations between
  - bit-vector variable and boolean variables
  - boolean variables and the numbers in pysat CNF
"""
from enum import Enum
import logging

import z3
from z3.z3util import get_vars

from .mapped_blast import translate_smt2formula_to_cnf
from ..bool.interpolant.core_based_itp import Z3Interpolant

logger = logging.getLogger(__name__)

"""
Bit-Vector Interpolant
"""


class ITPStrategy(Enum):
    FLATTENING = 0


class BVInterpolant:

    def __init__(self):
        self.fml = None
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat?
        self.vars = []
        self.verbose = 0