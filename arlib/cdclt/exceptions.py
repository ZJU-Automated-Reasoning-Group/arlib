"""Exceptions used by the CDCL(T) SMT solver
NOTE: we also use exceptions to indicate some "good states", e.g., the theory
  solver decides that a Boolean model is $T$-satisfiable.
"""
from arlib.utils.exceptions import SMTSuccess, SMTError


class TheorySolverSuccess(SMTSuccess):
    """
    The theory solver checks T-consistency successfully (the Boolean model
    is T-consistency), which means the original SMT formula is satisfiable
    """


class TheorySolverError(SMTError):
    """
    The theory solver encounters an error while checking T-consistency
    """


class BoolSolverSuccess(SMTSuccess):
    """
    The Boolean solver checks satisfiability successfully
    In the CDDL(T) settings, it means that the Boolean skeleton is unsatisfiable (
    Thus, the original SMT formula is also unsatisfiable
    )
    """


class BoolSolverError(SMTError):
    """
    The Bool solver encounters an error when sampling Boolean models
    """


class SimplifierSuccess(SMTSuccess):
    """
    If the pre-processing phase can decide the satisfiability of the formula (SAT or UNSAT)
    Then, we can terminate the solving process.
    """


class SimplifierError(SMTError):
    """
    The simplifier/pre-processing phase encounters an error
    """


class PySMTSolverError(SMTError):
    """
    PySMT solver encounters an error while checking T-consistency
    """
