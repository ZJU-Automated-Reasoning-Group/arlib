"""
Stålmarck-style symbolic abstraction (framework + simple propositional instantiation).

This module provides an extensible implementation of the algorithms described in
"A Method for Symbolic Computation of Abstract Operations" (Thakur & Reps).

The implementation focuses on three components:
- Integrity-constraint generation over fresh Boolean variables for sub-formulas
- 0-assume: local propagation until a fixpoint (anytime, monotone from above)
- k-assume: recursive branch-and-merge (Dilemma Rule) up to depth k
"""
