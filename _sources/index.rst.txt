Welcome to arlib's Documentation!
=================================


=============
Introduction
=============

Arlib is a toolkit for playing with various automated reasoning tasks.  Some of its key features include:

* Abductive inference (``arlib/abduction``)
* AllSMT (``arlib/allsmt``)
* Backbone computation (``arlib/backbone``)
* UNSAT core extraction (``arlib/unsat_core``)
* Exists-forall SMT formulas (``arlib/quant``)
* General quantified SMT formulas (``arlib/quant``)
* Quantifier elimination (``arlib/quant/qe``)
* Sampling solutions of SMT formulas (``arlib/sampling``)
* Counting the models of SMT formulas (``arlib/counting``)
* Optimization Modulo Theory (OMT) (``arlib/optimization``)
* Interpolant generation (``arlib/bool/interpolant``)
* Symbolic abstraction (``arlib/symabs``)
* Predicate abstraction (``arlib/symabs/predicate_abstraction``)
* Monadic predicate abstraction (``arlib/monabs``)
* Knowledge compilation (``arlib/bool/knowledge_compiler``)
* (Weighted) MaxSAT (``arlib/bool/maxsat``)
* QBF solving
* Finite Field Solving (``arlib/smt/ff``)
* Formula rewritings/simplifications
* Interactive theorem proving (``arlib/itp``)
* LLM integration (``arlib/llm``)
* Automata operations (``arlib/automata``)
* SyGuS (Syntax-Guided Synthesis) (``arlib/sygus``)
* ...

We welcome any feedback, issues, or suggestions for improvement. Please feel free to open an issue in our repository.

==========================
Installing and Using Arlib
==========================

Install arlib from source
---------------------------------------

::

  git clone https://github.com/ZJU-Automated-Reasoning-Group/arlib
  virtualenv --python=python3 venv
  source venv/bin/activate
  cd arlib
  bash setup_local_env.sh
  pip install -e .

The setup script will:
- Create a Python virtual environment if it doesn't exist
- Activate the virtual environment and install dependencies from requirements.txt
- Download required solver binaries (CVC5, MathSAT, z3)
- Run unit tests if available

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   topics
   abduction
   backbone
   cdclt
   counting
   ff
   interpolant
   itp
   knowledge_compilation
   monabs
   optimization
   quantifiers
   sampling
   smt
   symbolic_abstraction
   predicate_abstraction
   unsat_core
   allsmt
   applications
