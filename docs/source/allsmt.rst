===================
allsmt
===================


Introduction
=========

Related Work
=========

A closely relate problem is AllSAT, which can be categorized into
two main types: blocking solvers and non-blocking solvers.

- Blocking AllSAT solvers  are built on top of Conflict-Driven
Clause Learning (CDCL) and non-chronological backtracking (NCB).
- Non-blocking AllSAT solvers [10, 15] address the inefficiencies associated with blocking clauses by avoiding their use altogether. Instead, these
solvers employ chronological backtracking (CB)



allsmt in Arlib
=========


Usage Examples
-------------

.. code-block:: python

    from arlib import allsmt
    # TBD



References
=========

- TACAS 05: H. Jin, H. Han, F. Somenzi, Efficient conflict analysis for finding all satisfying assignments of a Boolean circuit
- CAV 02: K. L. McMillan, Applying SAT methods in unbounded symbolic model
checking.
- FMCAD 04: O. Grumberg, A. Schuster, A. Yadgar, Memory efficient all-solutions
SAT solver and its application for reachability analysis.
- DATE 04: B. Li, M. S. Hsiao, S. Sheng, A novel SAT all-solutions solver for efficient
preimage computation
- Disjoint Projected Enumeration for SAT and SMT
without Blocking Clauses. Giuseppe Spallitta,  Roberto Sebastiani,
Armin Biere. https://github.com/giuspek/tabularAllSAT
