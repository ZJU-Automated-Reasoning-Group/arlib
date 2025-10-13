Research Topics and Thesis Projects
===================================

Arlib offers numerous opportunities for research and thesis projects across multiple areas of automated reasoning.

=========
Core Algorithm Development
=========

**Parallel CDCL(T) Solving** (``arlib/smt/pcdclt``)
  Develop parallel algorithms for conflict-driven clause learning with theory reasoning. Focus on work distribution, clause sharing, and portfolio solving.

**Optimization Modulo Theory** (``arlib/optimization``)
  Extend SMT solving with optimization capabilities. Implement algorithms for OMT over bit-vectors, arithmetic, and mixed theories.

**Advanced Model Counting** (``arlib/counting``)
  Improve counting algorithms for Boolean, arithmetic, and quantifier-free bit-vector formulas. Focus on scalability and approximation techniques.

**Symbolic Abstraction** (``arlib/symabs``)
  Develop new abstraction techniques for infinite state systems. Implement counterexample-guided abstraction refinement (CEGAR).

=========
Theory-Specific Solving
=========

**Finite Field SMT** (``arlib/smt/ff``)
  Build decision procedures for Galois field constraints. Applications in cryptography and coding theory.

**Floating-Point Arithmetic** (``arlib/smt/fp``)
  Develop efficient solvers for IEEE 754 floating-point constraints with proper handling of rounding modes and special values.

**String Constraint Solving**
  Extend string theory support with automata-based techniques. Implement length constraints and regular language operations.

=========
AI-Enhanced Reasoning
=========

**LLM-Driven Constraint Solving** (``arlib/llm``)
  Integrate large language models to guide solver heuristics, strategy selection, and formula preprocessing.

**Machine Learning for Solvers** (``arlib/ml``)
  Extract features for learned solver selection, clause learning prediction, and variable ordering heuristics.

**Automata Learning** (``arlib/automata``)
  Apply active learning to infer automata from examples for string constraint solving and program verification.

=========
Advanced Sampling & Enumeration
=========

**Uniform Sampling** (``arlib/sampling``)
  Develop algorithms for uniform solution sampling over complex constraint domains. Applications in probabilistic verification.

**AllSMT Algorithms** (``arlib/allsmt``)
  Enumerate all solutions efficiently. Focus on diversity metrics and incremental solving techniques.

**Solution Space Analysis**
  Implement tools for analyzing solution spaces, including backbone computation and minimal unsatisfiable core extraction.

=========
Quantifier Handling
=========

**Quantifier Elimination** (``arlib/quant/qe``)
  Develop QE procedures for mixed theories combining arithmetic, bit-vectors, and arrays.

**E-Matching Optimization** (``arlib/quant/ematching``)
  Improve quantifier instantiation through better pattern matching and trigger selection.

**CHC Solving** (``arlib/quant/chctools``)
  Scale algorithms for constrained Horn clause solving. Applications in program verification and synthesis.

=========
Applications & Tools
=========

**Interactive Theorem Proving** (``arlib/itp``)
  Build proof assistant tools with support for multiple theories and automated proof search.

**Program Synthesis** (``arlib/synthesis``)
  Implement syntax-guided synthesis techniques for bit-vectors, arithmetic, and string domains.

**Abductive Reasoning** (``arlib/abduction``)
  Develop algorithms for generating explanations and hypotheses from constraint observations.

=========
Getting Started
=========

Each module includes examples and documentation. Start with ``arlib/allsmt`` for basic usage patterns, then explore specialized areas based on your interests.
