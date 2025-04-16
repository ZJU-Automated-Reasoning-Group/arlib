SMT Solving for Finite Field
============================

.. _ff-solving:

================
Overview
================


Finite filed arithmetic is a fundamental component of many cryptographic protocols.


Recent works have significantly advanced SMT solving for finite fields, including:

* MCSat-based techniques in Yices2
* Split Gröbner Bases approaches
* Non-linear reasoning methods

================
Implementation
================

Arlib provides a flexible implementation for SMT solving over finite fields in the ``arlib/smt/ff`` module. It supports multiple encoding strategies:

* **Bit-vector encoding** (``ff_bv_solver.py``): Translates finite field operations to bit-vector constraints
* **Integer encoding** (``ff_int_solver.py``): Uses modular arithmetic via integer constraints

The core components include:

* Formula parsing and representation (``ff_parser.py``)
* Translation to bit-vector formulas using Z3
* Translation to integer arithmetic with modular operations
* Support for standard finite field operations (addition, multiplication, equality)


**Bit-vector Encoding**

The bit-vector approach represents finite field elements as fixed-width bit-vectors:

* Field elements are encoded as bit-vectors with width log₂(field_size)
* Field operations are translated to bit-vector operations with modular constraints
* Range constraints ensure values stay within the field bounds

**Integer Encoding**

The integer encoding represents field elements as integers with modular arithmetic:

* Field elements are constrained to the range [0, field_size-1]
* Field operations are translated to integer operations with explicit modulo operations
* This approach leverages integers and non-linear arithmetic reasoning in SMT solvers

================
Usage Example
================

Here's a simple example of using the finite field solver:

.. code-block:: python

    from arlib.smt.ff.ff_bv_solver import solve_qfff
    
    # Example formula in SMT-LIB format
    smt_input = """
    (set-info :smt-lib-version 2.6)
    (set-logic QF_FF)
    (declare-fun x () (_ FiniteField 17))
    (declare-fun y () (_ FiniteField 17))
    (assert (= (ff.add x y) #f3m17))
    (assert (= (ff.mul x y) #f5m17))
    (check-sat)
    """
    
    # Solve the formula
    solve_qfff(smt_input)

This will solve the system of equations x + y = 3 and x * y = 5 in the finite field GF(17).

References
==========

* Hader, T., Kaufmann, D., Irfan, A., Graham-Lengrand, S., & Kovács, L. (2024).
    `MCSat-based Finite Field Reasoning in the Yices2 SMT Solver`_.
    ArXiv:2402.17927.

* Ozdemir, A., Pailoor, S., Bassa, A., Ferles, K., Barrett, C., & Dillig, I. (2024).
    Split Gröbner Bases for Satisfiability Modulo Finite Fields.
    In *Computer Aided Verification (CAV)*.

* Ozdemir, A., Kremer, G., Tinelli, C., & Barrett, C. (2023).
    Satisfiability modulo finite fields.
    In *Computer Aided Verification (CAV)*.

* Hader, T., Kaufmann, D., & Kovács, L. (2023).
    SMT solving over finite field arithmetic.
    In *Logic for Programming, Artificial Intelligence and Reasoning (LPAR)*.


* Hader, T. (2022).
    Non-linear SMT-reasoning over finite fields.
    Master's thesis, TU Wien.

* Hader, T., & Kovács, L. (2022).
    Non-linear SMT-reasoning over finite fields.
    In *SMT Workshop*. Extended Abstract.

.. _MCSat-based Finite Field Reasoning in the Yices2 SMT Solver: https://arxiv.org/pdf/2402.17927
