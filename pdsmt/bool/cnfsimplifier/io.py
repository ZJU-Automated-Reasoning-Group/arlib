# coding: utf-8
from typing import List
from .clause import Clause
from .cnf import Cnf
from .variable import Variable


class NumericClausesReader:
    """
    Build an internal CNF object from Boolean clauses [[int]]
    """
    def __init__(self):
        pass

    def read(self, clauses: List[List[int]]) -> Cnf:
        """
        "Parse" numerical clauses
        """
        clause_list = list()
        for cls in clauses:
            var_list = list()
            for var in cls:
                if var != 0:
                    var_list.append(Variable(var))
            clause_list.append(Clause(var_list))
        cnf = Cnf(clause_list)
        return cnf
