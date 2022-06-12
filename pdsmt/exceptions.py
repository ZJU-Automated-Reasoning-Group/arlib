# coding: utf-8
"""
Public subclasses of Exception
"""


class SMTError(Exception):
    pass


class ExecutorError(SMTError):
    pass


class SmtlibError(SMTError):
    pass


class Z3NotFoundError(SmtlibError):
    pass


class SolverError(SmtlibError):
    pass


class SolverUnknown(SolverError):
    pass
