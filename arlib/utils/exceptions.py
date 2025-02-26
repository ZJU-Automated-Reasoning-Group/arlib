# coding: utf-8
"""
Public subclasses of different Exceptions
"""


class ArlibException(Exception):
    """Base class for Arlib exceptions"""
    pass


class SMTSuccess(ArlibException):
    """Flag for good state"""
    pass


class SMTError(ArlibException):
    """TBD"""
    pass


class SMTUnknown(ArlibException):
    """TBD"""
    pass


class SMTLIBSolverError(SMTError):
    """TBD"""
    pass


class UndefinedLogicError(ArlibException):
    """This exception is raised if an undefined Logic is attempted to be used."""
    pass


class NoLogicAvailableError(ArlibException):
    """Generic exception to capture errors caused by missing support for logics."""
    pass

