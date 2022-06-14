# coding: utf-8
"""
See https://theory.stanford.edu/~nikolaj/programmingz3.html#sec-consequences
a, b, c, d = Bools('a b c d')
s = Solver()
s.add(Implies(a, b), Implies(c, d))   # background formula
print(s.consequences([a, c],          # assumptions
                     [b, c, d]))      # what is implied?
produces the result:
   (sat, [Implies(c, c), Implies(a, b), Implies(c, d)])

  TODO: can we use the consequence finding facility for reducing the search space? The idea:
     One of the key problem in CDCL(T) is to "learn" the correlations between the abstracted atoms.
     Usually, the theory solver can find the complicated correlations gradually, which are feed to
     the Boolean solver implicitly.
     However, can we find the correlations in the preprocessing step? (so that the Boolean solver can
     have more knowledge about the original SMT formula). In general, this problem can be quite challenging.
     But, what if we focus on the pair-wise and context-independent correlations:
        - **Pair-wise**: only care about the relations between two atoms A and B
        - **Context-independent**: do not take the "whole background SMT" into consideration. We only encode
         the information (A = Atom_A, B = Atom_B, ...), and then use the consequence finding facility to find
         the correlations between A, B,...

 Related work:
    - Relevancy propagation in Z3.
    - Theory-aware branching
"""

import logging
from typing import List
import z3


class ConsequenceFinder(object):
    # FIXME

    def __init__(self):
        self.solver = z3.Solver()

    def add(self, exp):
        self.solver.add(exp)

    def get_consequences(self, set_a: List[z3.BoolRef], set_b: List[z3.BoolRef]):
        try:
            res, facts = self.solver.consequences([set_a], [set_b])
            if res == z3.sat:
                return facts
        except Exception as ex:
            raise ex
