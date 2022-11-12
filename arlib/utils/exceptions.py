# coding: utf-8
"""
Public subclasses of different Exceptions for the arlib library

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
    """


class ForAllSolverSuccess(SMTSuccess):
    """
    TBD
    """


class SimplifierSuccess(SMTSuccess):
    """
    TBD
    """


class SMTUnknown(Exception):
    """
    TBD
    """


class ExitsSolverUnknown(SMTUnknown):
    """
    """


class ForAllSolverUnknown(SMTUnknown):
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
