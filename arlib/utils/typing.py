# coding: utf-8
"""
Common types for different components of arlib
"""
from enum import Enum


class SolverResult(Enum):
    UNSAT = -1
    UNKNOWN = 0
    SAT = 1
    ERROR = 2


class OSType(Enum):
    LINUX = 0
    WINDOWS = 1
    MAC = 2
    UNKNOWN = 3
