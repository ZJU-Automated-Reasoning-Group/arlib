Applications
============

Arlib supports various applications across testing, verification, synthesis, and optimization.

==========
Testing
==========

**Constrained Random Testing**
  Generate test cases satisfying logical constraints using ``arlib/sampling`` and ``arlib/allsmt``.

**Combinatorial Testing**
  Generate diverse test suites with ``arlib/bool/features`` for covering parameter interactions.

==========
Verification
==========

**Predicate Abstraction**
  Abstract program states using ``arlib/symabs/predicate_abstraction`` for verification.

**Symbolic Abstraction**
  Abstract infinite state spaces with ``arlib/symabs`` for model checking.

**Interactive Theorem Proving**
  Formal verification with ``arlib/itp`` framework supporting multiple theories.

==========
Synthesis
==========

**Program Synthesis**
  Synthesize programs from specifications using ``arlib/synthesis`` (SyGuS, PBE).

**Syntax-Guided Synthesis**
  Generate programs matching given grammars with ``arlib/synthesis/sygus_*``.

==========
Optimization
==========

**Optimization Modulo Theory**
  Solve optimization problems over logical theories using ``arlib/optimization``.

**MaxSAT Solving**
  Solve maximum satisfiability problems with ``arlib/bool/maxsat``.

==========
Learning & AI
==========

**LLM-Enhanced Reasoning**
  Integrate large language models with ``arlib/llm`` for constraint solving.

**Machine Learning Features**
  Extract features for ML-based solver selection with ``arlib/ml``.
