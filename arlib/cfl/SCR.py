#!pypy3
import sys
import cProfile
from typing import List

from arlib.cfl.graph import Graph
from arlib.cfl.grammar import Grammar
from arlib.cfl.sc_solver import SCSolver


def main(argv: List[str]) -> None:
    arg: List[str] = ['demo/200KB.dot','demo/VM_Grammar.txt','Matrix','Cubic']
    arg[0] = sys.argv[1]
    print('SC start processing', arg[0])
    graph = Graph(arg[0],arg[2])
    grammar = Grammar(arg[1])
    solver = SCSolver(arg[3])
    solver.solve(graph, grammar)
    print('complete', arg[0])


if __name__ == '__main__':
    main(sys.argv)
