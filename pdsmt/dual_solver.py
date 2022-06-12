# coding: utf-8

"""
Dual propagation and implicants for CDCL(T):

     The propositional assignment produced by prop is not necessarily minimal.
     It may assign truth assignments to literals that are irrelevant to truth of the set of clauses.

     To extract a smaller assignment, one trick is to encode the negation of the clauses in a separate
     dual solver.

     A truth assignment for the primal solver is an unsatisfiable core for the dual solver.
"""


def enumerate_unsat_cores():
    return 0
