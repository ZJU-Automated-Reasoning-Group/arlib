# coding: utf-8
"""
Public subclasses of Exception
"""


class SMTSuccess(Exception):
    pass


class TheorySolverSuccess(SMTSuccess):
    pass


class BoolSolverSuccess(SMTSuccess):
    pass


class SimplifierSuccess(SMTSuccess):
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
