# coding: utf-8
"""
Public subclasses of Exception
"""


class TheorySolverSuccess(Exception):
    pass


class SMTError(Exception):
    pass


class ExecutorError(SMTError):
    pass


class SmtlibError(SMTError):
    pass


class SolverError(SmtlibError):
    pass


class SolverUnknown(SolverError):
    pass
