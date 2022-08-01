# coding: utf-8
"""
Public subclasses of different Exceptions for the pdsmt library

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


class ExitsSolverSuccess(SMTSuccess):
    """
    The theory solver checks T-consistency successfully
    """


class ForAllSolverSuccess(SMTSuccess):
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
