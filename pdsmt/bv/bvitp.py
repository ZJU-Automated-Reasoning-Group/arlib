# coding: utf-8
"""
1. Translate a bit-vector optimization problem to a weighted MaxSAT problem,
2. Call a third-path MaxSAT solver

TODO:
- Need to track the relations between
  - bit-vector variable and boolean variables
  - boolean variables and the numbers in pysat CNF
"""
import logging
import random
import time
from typing import List

import z3
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from z3.z3util import get_vars

from .mapped_blast import translate_smt2formula_to_cnf
from .maxsat import MaxSATSolver

logger = logging.getLogger(__name__)

"""
Bit-Vector Interpolant
"""