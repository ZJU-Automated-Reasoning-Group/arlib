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
logger = logging.getLogger(__name__)

"""
Bit-Vector Interpolant
"""