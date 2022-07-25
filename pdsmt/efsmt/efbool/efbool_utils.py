from enum import Enum

sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41',
               'lingeling', 'maplechrono', 'maplecm', 'maplesat', 'minicard',
               'mergesat3', 'minisat22', 'minisat-gh']


class EFBoolResult(Enum):
    """Result of EFBool Checking"""
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3
