# coding: utf-8
"""
Public subclasses of different Exceptions

"""


class SMTSuccess(Exception):
    """
    Flag for good state
    """


class SMTError(Exception):
    """
    TBD
    """


class TheorySolverSuccess(SMTSuccess):
    """
    The theory solver checks T-consistency successfully
    """


class TheorySolverError(SMTError):
    """
    """


class BoolSolverSuccess(SMTSuccess):
    """
    TBD
    """


class BoolSolverError(SMTError):
    """
    """


class SimplifierSuccess(SMTSuccess):
    """
    TBD
    """


class SimplifierError(SMTError):
    """
    """


class PySMTSolverError(SMTError):
    """
    TBD
    """


class SMTLIBSolverError(SMTError):
    """
    TBD
    """

