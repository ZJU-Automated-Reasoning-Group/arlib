# coding: utf-8
from .variable import Variable
from .cnf import Cnf
from .clause import Clause


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
