# coding: utf-8
"""
Small demo showing how to run Dissolve on a CNF.
"""
import logging
from pysat.formula import CNF

from arlib.bool.dissolve import Dissolve, DissolveConfig


def main():
    logging.basicConfig(level=logging.INFO)
    # Example CNF: (x1 or x2) & (not x1 or x3) & (x2 or x3)
    cnf = CNF(from_clauses=[[2], [-1, 3], [2, 3]])

    solver = Dissolve(DissolveConfig(k_split_vars=2, per_query_conflict_budget=5000, max_rounds=10))
    res = solver.solve(cnf)
    print(res)


if __name__ == "__main__":
    main()
