===================
Interpolant Generation
===================


Introduction to Interpolation
=========


Given two contradictory formulas ``A`` and ``B``, a Craig interpolant ``I`` is a formula that satisfies the following conditions:

+ ``I`` is a logical consequence of ``A``.
+ ``I`` and ``B`` are contradictory
+ ``I`` contains only the variables that are common to ``A`` and ``B``.

Craig interpolant is a formula that captures the shared information
between ``A`` and ``B``, and "explains" why they are contradictory.
Craig interpolants have several important applications in model checking

- Abstraction refinement in verification
- Approximating the image computation in verification
- ...?

The computation of Craig interpolants is a challenging problem, and various 
algorithms and techniques have been developed to compute them efficiently. 

Interpolant Generation in Arlib
=========

References
=========

- McMillan, K. L. (2003). "Interpolation and SAT-based Model Checking." In Proceedings of the International Conference on Formal Methods in Computer-Aided Design (FMCAD)