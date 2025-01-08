
AllSMT
==========================

==============
Introduction
==============

AllSMT is a problem that generalizes AllSAT to SMT formulas. Given an SMT formula, the goal is to find all satisfying assignments. This problem is important in many applications, such as combinational iteeration testing, quantitative program analysis,
and constrained test generation.


==============
Related Work
==============

A closely relate problem is AllSAT. The AllSAT problem is NP-hard, and it is known that the problem is not fixed-parameter tractable unless P=NP.
Existing AllSAT solvers can be categorized into
two main types: blocking solvers and non-blocking solvers.

- Blocking AllSAT solvers  are built on top of Conflict-Driven Clause Learning (CDCL) and non-chronological backtracking (NCB).

- Non-blocking AllSAT solvers address the inefficiencies associated with blocking clauses by avoiding their use altogether. Instead, these solvers employ chronological backtracking (CB)

===============
allsmt in Arlib
===============



==============
References
==============

- **SAT '23**: Gabriele Masina, Giuseppe Spallita, Roberto Sebastiani. *On CNF Conversion for Disjoint SAT Enumeration*.
- **TACAS '05**: H. Jin, H. Han, F. Somenzi. *Efficient Conflict Analysis for Finding All Satisfying Assignments of a Boolean Circuit*.
- **CAV '02**: K. L. McMillan. *Applying SAT Methods in Unbounded Symbolic Model Checking*.
- **FMCAD '04**: O. Grumberg, A. Schuster, A. Yadgar. *Memory Efficient All-Solutions SAT Solver and Its Application for Reachability Analysis*.
- **DATE '04**: B. Li, M. S. Hsiao, S. Sheng. *A Novel SAT All-Solutions Solver for Efficient Preimage Computation*.
- *Disjoint Projected Enumeration for SAT and SMT without Blocking Clauses*. Giuseppe Spallitta, Roberto Sebastiani, Armin Biere. `GitHub Repository <https://github.com/giuspek/tabularAllSAT>`_.