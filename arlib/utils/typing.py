# coding: utf-8
"""
Common types for different components of arlib
"""
from enum import Enum, auto


class SolverResult(Enum):
    UNSAT = auto()
    UNKNOWN = auto()
    SAT = auto()
    ERROR = auto()


class OSType(Enum):
    LINUX = auto()
    WINDOWS = auto()
    MAC = auto()
    UNKNOWN = auto()