from typing import Union, List

import z3


def int_clauses_to_z3(clauses: List[List[int]]) -> z3.z3.BoolRef:
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
