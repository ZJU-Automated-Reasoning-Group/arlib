Summer Research, Honours/Master Thesis Project Topics
==================


=========
Beyond SMT Solving
=========



Parallel Bit-Vector Optimizations
-------

Optimization Modulo Theory (OMT) is an extension of SMT, which is used for checking the 
satisfiability of logical formulas with respect to background theories such as 
arithmetic, arrays, and bit vectors. 
OMT extends this by adding optimization capabilities, enabling it to find solutions 
that minimize or maximize a given objective function.

Here, we are interested in OMT(BV) problems, where the solution space is characterized by a
quantifier-free bit-vector formula.
Please refer to `arlib/bv/bvopt.py` for single-objective and multi-objectives optimization.

(In some algorithms, we may reduce a single-objective optimization problem to a special 
multi-objectives optimization problem (e.g., "Bit-vector optimization, TACAS'16"))

Bit-Vector Interpolation
-------

Given two contradictory formulas `A` and `B`, a Craig interpolant `I` is a formula that satisfies the following conditions:
- `I` is a logical consequence of `A`.
- `I` and `B` are contradictory
- `I` contains only the variables that are common to `A` and `B`.

Please refer to `arlib/bv/bvitp.py`.


Bit-Vector Model Counting
-------

Model counting is the problem of determining the number of possible solutions 
(models) to a given formula. 


Refer to `arlib/bv/bv_counting`.

Bit-Vector Model Sampling
-------

=========
SMT Solving for Specific Theories
=========


SMT Solving for Galois Field
--------

A Galois Field, also known as a finite field, is a mathematical structure that 
consists of a finite set of elements and two operations, typically addition 
and multiplication. Galois Fields are used in many areas of mathematics, 
computer science, and engineering, such as coding theory, cryptography, and 
digital signal processing.


SMT Solving for Exists-Forall Problems
--------



=========
References
=========
