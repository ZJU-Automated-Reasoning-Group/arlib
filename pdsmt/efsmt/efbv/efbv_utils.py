from enum import Enum


class EFBVResult(Enum):
    """Result of EFBV checking"""
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3


class EFBVTactic(Enum):
    """Tactics for solving Exists-ForAll problems over bit-vectors"""
    Z3_QBF = 0
    Z3_BV = 1
    EXTERNAL_QBF = 2
    EXTERNAL_BV = 3
    SIMPLE_CEGAR = 4
