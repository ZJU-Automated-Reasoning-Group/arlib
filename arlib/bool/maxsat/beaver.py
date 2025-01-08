"""
MaxSAT with bit-vector optimization based the following paper
 SAT 2018: A. Nadel. Solving MaxSAT with bit-vector optimization. ("Mrs. Beaver")
https://www.researchgate.net/profile/Alexander-Nadel/publication/325970660_Solving_MaxSAT_with_Bit-Vector_Optimization/links/5b3a08fb4585150d23ee95df/Solving-MaxSAT-with-Bit-Vector-Optimization.pdf

Related Work:
FMCAD 19: Anytime Weighted MaxSAT with Improved Polarity Selection and Bit-Vector Optimization
https://theory.stanford.edu/~barrett/fmcad/papers/FMCAD2019_paper_16.pdf
"""

from pysat.solvers import Solver


