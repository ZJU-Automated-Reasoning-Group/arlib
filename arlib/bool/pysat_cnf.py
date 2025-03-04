"""CNF manipulation via pysat"""
import sys
import random
from copy import deepcopy
from typing import List
from pysat.formula import CNF  # IDPool
from pysat.solvers import Solver


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


def check_assumption_literals(fml: CNF, assumption_literals: List[int]) -> bool:
    """
    Check if the assumption literals are consistent with the formula
    """
    solver = Solver(bootstrap_with=fml)
    return solver.solve(assumptions=assumption_literals)



def demo_gen_cubes():
    """test case"""
    clauses = [[1, 2], [4, 5], [-1, 2, 3]]
    fml = CNF(from_clauses=clauses)
    print(gen_cubes(fml, 3))


if __name__ == "__main__":
    demo_gen_cubes()