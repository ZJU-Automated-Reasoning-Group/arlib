# coding: utf-8
"""
Public subclasses of Exception
"""


class SMTSuccess(Exception):
    """
    Flag for good state
    """


class TheorySolverSuccess(SMTSuccess):
    """
    The theory solver checks T-consistency successfully
    """


class BoolSolverSuccess(SMTSuccess):
    """
    TBD
    """


class SimplifierSuccess(SMTSuccess):
    """
    TBD
    """


class SMTError(Exception):
    """
    TBD
    """


class PySMTSolverError(SMTError):
    """
    TBD
    """


class SMTLIBSolverError(SMTError):
    """
    TBD
    """
