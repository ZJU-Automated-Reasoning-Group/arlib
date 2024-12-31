Predicate Abstraction
====================

Introduction to Predicate Abstraction
-----------------------------------

Predicate abstraction is a technique for constructing finite-state abstractions
from large or infinite-state systems.

The fundamental operation in predicate abstraction can be summarized as follows:
Given a formula φ and a set of predicates P in a theory T, generate the most
precise approximation of φ using P.

Depending on the nature of the problem domain, one may either want to generate:

(i) the best underapproximation of φ, i.e., the weakest Boolean combination
    of P that implies φ (denoted by FP(φ)) or

(ii) the best overapproximation of φ, i.e., the strongest Boolean combination of P
     that is implied by φ (denoted by GP(φ)).

Here, the notions of weakness, strength and implication are with respect
to entailment in the given theory T.

Predicate Abstraction in Arlib
-----------------------------


References
----------

- [GS97] S. Graf and H. Saidi. Construction of abstract state graphs with PVS. In
  CAV'97
- [CKSY04] E. Clarke, D. Kroening, N. Sharygina, and K. Yorav. Predicate abstraction
  of ANSI–C programs using SAT. FMSD'04
- [LBC03] S. K. Lahiri, R. E. Bryant, and B. Cook. A symbolic approach to predicate
  abstraction. In CAV'03
- [LB04] S. K. Lahiri and R. E. Bryant. Constructing Quantified Invariants via
  Predicate Abstraction. In VMCAI'04

