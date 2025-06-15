Summer Research, Honours/Master Thesis Project Topics
==================


=========
1. Beyond SMT Solving
=========



Parallel Bit-Vector Optimizations
-------

Optimization Modulo Theory (OMT) is an extension of SMT, which is used for checking the 
satisfiability of logical formulas with respect to background theories such as 
arithmetic, arrays, and bit vectors. 
OMT extends SMT by adding optimization capabilities, enabling it to find solutions
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

Given a satisfiable formula `P`, how to generate multiple and diverse solutions `P`?

Parallel CDCL(T) Solving
-------

Develop parallel algorithms for Conflict-Driven Clause Learning with theories.
Focus on efficient work distribution and conflict clause sharing across multiple threads.

Refer to `arlib/smt/pcdclt`.

Symbolic Abstraction and Refinement
-------

Investigate predicate abstraction techniques and counterexample-guided abstraction refinement (CEGAR).
Applications include program verification and model checking.

Refer to `arlib/symabs`.

=========
2. SMT Solving for Specific Theories
=========

SMT Solving for String Constraints
--------

(We have an idea about parallel string constraint solving)

SMT Solving for Galois Field
--------

A Galois Field, also known as a finite field, is a mathematical structure that 
consists of a finite set of elements and two operations, typically addition 
and multiplication. Galois Fields are used in many areas of mathematics, 
computer science, and engineering, such as coding theory, cryptography, and 
digital signal processing.

Refer to `arlib/smt/ff`.

SMT Solving for Exists-Forall Problems over Bit-Vectors
--------

SMT Solving for Floating-Point Arithmetic
--------

Develop efficient decision procedures for IEEE 754 floating-point constraints.
Focus on rounding modes, special values (NaN, infinity), and precision handling.

Refer to `arlib/smt/fp`.

=========
3. Learning and AI-Enhanced Reasoning
=========

LLM-Driven Constraint Solving
--------

Integrate large language models to guide SMT solver heuristics and strategy selection.
Explore neural-symbolic approaches for automated reasoning.

Refer to `arlib/llm/smto`.

Automata Learning for Constraint Solving
--------

Apply active learning techniques to infer finite automata and symbolic finite automata.
Applications in string constraint solving and program verification.

Refer to `arlib/automata`.

LLM-Based Abductive Reasoning
--------

Use language models to generate explanations and hypotheses for observed constraints.
Focus on debugging and root cause analysis in constraint systems.

Refer to `arlib/llm/abduct`.

=========
4. Advanced Sampling and Enumeration
=========

Uniform Sampling for Linear Integer Arithmetic
--------

Develop algorithms for generating uniformly distributed solutions over linear integer constraints.
Applications in testing and probabilistic verification.

Refer to `arlib/sampling/linear_ira`.

Non-Linear Real Arithmetic Sampling
--------

Efficient sampling techniques for polynomial constraints over real numbers.
Focus on volume computation and density estimation.

Refer to `arlib/sampling/nonlinear_ira`.

All-SMT and Solution Enumeration
--------

Enumerate all satisfying assignments or a diverse subset of solutions.
Applications in combinatorial optimization and test case generation.

Refer to `arlib/allsmt`.

=========
5. Quantifier Reasoning
=========

Quantifier Elimination for Mixed Theories
--------

Develop efficient QE procedures for combinations of arithmetic, bit-vectors, and arrays.
Focus on applications in program verification and synthesis.

Refer to `arlib/quant/qe`.

E-Matching and Instantiation Strategies
--------

Improve quantifier instantiation in SMT solvers through better pattern matching
and trigger selection heuristics.

Refer to `arlib/quant/ematching`.

Constrained Horn Clause Solving
--------

Develop scalable algorithms for solving systems of constrained Horn clauses.
Applications in program verification and invariant synthesis.

Refer to `arlib/quant/chctools`.

=========
References
=========
