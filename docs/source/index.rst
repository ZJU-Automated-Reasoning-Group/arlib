Welcome to arlib's Documentation!
=================================


=============
Introduction
=============

Arlib is a toolkit for playing with various automated reasoning tasks.  Some of its key features include:

* Solving exits-forall SMT formulas (``arlib/quant``)
* Solving general quantified SMT formulas (``arlib/quant``)
* Quantifier elimination (``arlib/quant/qe``)
* Sampling solutions of SMT formulas (``arlib/smt/sampling``)
* Counting the models of SMT formulas (``arlib/smt/bv/qfbv_counting``)
* Optimization Modulo Theory (OMT) solving (``arlib/optimization``)
* Interpolant generation (``arlib/bool/interpolant'')
* Minimal satisfying assignment (``arlib/optimization``)
* Symbolic abstraction (``arlib/symabs'')
* Abductive inference (``arlib/abduction``)
* Backbone (``arlib/backbone``)
* Knowledge compilation (``arlib/bool/knowledge_compiler``)
* (Weighted) MaxSAT (``arlib/bool/maxsat``)
* QBF solving
* Formula rewritings/simplifications
* ...

We welcome any feedback, issues, or suggestions for improvement. Please feel free to open an issue in our repository.

=============
Installing and Using Arlib
=============

(TODO) Install arlib as a package
--------

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
   interpolant
   knowledge_compilation
   monabs
   optimization
   quantifiers
   sampling
   smt
   symbolic_abstraction
