"""CNF manipulation via pysat"""
import sys
import random
from copy import deepcopy
from typing import List
from pysat.formula import CNF  # IDPool


def simplify_cnf(fml: CNF, assumptions: List[int]) -> CNF:
    """ given a formula, return a new formula simplified by the assumptions"""
    result = list(filter(lambda cls: all(map(lambda lit: lit not in cls, assumptions)),
                         deepcopy(fml.clauses)))
    for cls in result:
        for lit in assumptions:
            try:
                cls.remove(-lit)
            except ValueError:
                continue

    return CNF(from_clauses=result)


def gen_cubes(fml: CNF, k_vars: int) -> List[List[int]]:
    """
    Randomly select k_vars variables and generate all cubes using those variables
    """
    total_vars = fml.nv
    assert k_vars <= total_vars
    cube_vars = random.sample([i + 1 for i in range(total_vars)], k_vars)
    all_cubes = []
    # Loop over all possible variable assignments
    for i in range(2 ** k_vars):
        cube = []
        # Convert the integer value to a list of Boolean values
        bool_list = [bool(i & (1 << j)) for j in range(k_vars)]
        # Assign each variable to its corresponding Boolean value
        for var_id in range(k_vars):
            cube.append(cube_vars[var_id] if bool_list[var_id] else -cube_vars[var_id])
        all_cubes.append(cube)

    return all_cubes


def demo_simplify_cnf():
    """test case"""
    clauses = [[1, 2], [4, 5], [-1, 2, 3]]
    fml = CNF(from_clauses=clauses)
    new_cnf = simplify_cnf(fml, [1, 4])
    new_cnf.to_fp(sys.stdout)
    print(gen_cubes(fml, 3))


if __name__ == "__main__":
    demo_simplify_cnf()
