# coding: utf-8
from enum import Enum


class Logic(Enum):
    QF_BOOL = 0,
    QF_BV = 1,
    QF_LRA = 2,
    QF_LIA = 3,
    QF_NRA = 4,
    QF_NIA = 5,
    QF_LIRA = 6,
    QF_ALL = 7


class BooleanStructure(Enum):
    """
    Supported Boolean Structure
    """
    ATOM = 0
    CONJUNCTION = 1,
    SMT_CNF = 2,
    SMT_DNF = 3,
    SMT_ANY = 4,
    BDD = 5,
    SDD = 6


class Sampler(object):
    # the supported logics for the Solver
    LOGICS = []

    def __init__(self, **options):
        return

    def sample(self, number=1):
        """
        Sample solutions
        """
        raise NotImplementedError
