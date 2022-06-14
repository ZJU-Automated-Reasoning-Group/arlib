from .simplifier import *
from .io import NumericClausesReader


def simplify_numeric_clauses(clauses):
    cnf = NumericClausesReader().read(clauses)
    new_cnf = subsumption_elimination(cnf)
    return new_cnf.get_numeric_clauses()