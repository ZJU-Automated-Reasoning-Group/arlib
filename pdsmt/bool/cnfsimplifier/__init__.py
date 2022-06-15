from .io import NumericClausesReader
from .simplifier import *


def simplify_numeric_clauses(clauses):
    cnf = NumericClausesReader().read(clauses)
    new_cnf = subsumption_elimination(cnf)
    return new_cnf.get_numeric_clauses()
