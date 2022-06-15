# coding: utf-8
from .clause import Clause
from .cnf import Cnf
from .variable import Variable


class NumericClausesReader:
    def __init__(self):
        pass

    def read(self, clauses):
        clause_list = list()
        for cls in clauses:
            var_list = list()
            for var in cls:
                if var != 0:
                    var_list.append(Variable(var))
            clause_list.append(Clause(var_list))
        cnf = Cnf(clause_list)
        return cnf
