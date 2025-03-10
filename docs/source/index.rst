Welcome to arlib's Documentation!
=================================


=============
Introduction
=============

Arlib is a toolkit for playing with various automated reasoning tasks.  Some of its key features include:

* Abductive inference (``arlib/abduction``)
* AllSMT (``arlib/allsmt``)
* Backbone (``arlib/backbone``)
* Exits-forall SMT formulas (``arlib/quant``)
* General quantified SMT formulas (``arlib/quant``)
* Quantifier elimination (``arlib/quant/qe``)
* Sampling solutions of SMT formulas (``arlib/smt/sampling``)
* Counting the models of SMT formulas (``arlib/smt/bv/qfbv_counting``)
* Optimization Modulo Theory (OMT) (``arlib/optimization``)
* Interpolant generation (``arlib/bool/interpolant``)
* Symbolic abstraction (``arlib/symabs``)
* Predicate abstraction (``arlib/symabs/predicate_abstraction``)
* Monadic predicate abstraction (``arlib/monabs``)
* Knowledge compilation (``arlib/bool/knowledge_compiler``)
* (Weighted) MaxSAT (``arlib/bool/maxsat``)
* QBF solving
* Finite Field Solving (`arlib/smt/ff`)
* Formula rewritings/simplifications
* ...

We welcome any feedback, issues, or suggestions for improvement. Please feel free to open an issue in our repository.

==========================
Installing and Using Arlib
==========================

(TODO) Install arlib as a package
---------------------------------------

::

  git cone https://github.com/ZJU-Automated-Reasoning-Group/arlib
  virtualenv --python=/usr/bin/...  venv
  source venv/bin/activate
  cd arlib
  python setup.py install




.. toctree::
   :maxdepth: 1
   :caption: Contents:

   topics
   abduction
   cdclt
   counting
   ff
   interpolant
   knowledge_compilation
   monabs
   optimization
   quantifiers
   sampling
   smt
   symbolic_abstraction
   predicate_abstraction
   allsmt
   applications
