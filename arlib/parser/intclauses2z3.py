from typing import Union, List

import z3


def int_clauses_to_z3(clauses: List[List[int]]) -> z3.BoolRef:
    """
    The int_clauses_to_z3 function takes a list of clauses, where each clause is a list of integers.
    The function returns the conjunction (AND) of all clauses in the input.
    Each integer represents an atomic proposition.
    :param clauses:List[List[int]]: Represent the clauses of a cnf
    :return: A z3 expr
    """
    z3_clauses = []
    vars = {}
    for clause in clauses:
        conds = []
        for t in clause:
            a = abs(t)
            if a in vars:
                b = vars[a]
            else:
                b = z3.Bool("k!{}".format(a))
                vars[a] = b
            b = z3.Not(b) if t < 0 else b
            conds.append(b)
        z3_clauses.append(z3.Or(*conds))
    return z3.And(*z3_clauses)
