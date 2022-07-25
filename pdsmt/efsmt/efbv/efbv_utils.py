from enum import Enum


class EFBVResult(Enum):
    """Result of EFBV Checking"""
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3


class EFBVTactic(Enum):
    Z3_QBF = 0
    Z3_BV = 1
    EXTERNAL_QBF = 2
    EXTERNAL_BV = 3
    SIMPLE_CEGAR = 4


sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41',
               'lingeling', 'maplechrono', 'maplecm', 'maplesat', 'minicard',
               'mergesat3', 'minisat22', 'minisat-gh']
