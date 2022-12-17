'''
Enumeration
NOTE: this is for Boolean!!
'''

from z3 import *

import time


def benchmark(name, function, param1, param2):
    """Benchmark a function with two parameters."""

    print('--' + name + ' approach--')
    start_time = time.perf_counter()
    print('Number of models: ' + str(function(param1, param2)))
    end_time = time.perf_counter()
    print('Time: ' + str(round(end_time - start_time, 2)) + ' s')


def count_models_with_solver(solver, variables):
    solver.push()  # as we will add further assertions to solver, checkpoint current state
    solutions = 0
    while solver.check() == sat:
        solutions = solutions + 1
        # Invert at least one variable to get a different solution:
        solver.add(Or([Not(x) if is_true(solver.model()[x]) else x for x in variables]))
    solver.pop()  # restore solver to previous state
    return solutions


import itertools


# Fastest enumeration: conditional checking.
def count_models_by_enumeration(solver, variables):
    solutions = 0
    for assignment in itertools.product(*[(x, Not(x)) for x in variables]):  # all combinations
        if solver.check(assignment) == sat:  # conditional check (does not add assignment permanently)
            solutions = solutions + 1
    return solutions


# Creating the assignment as a separate step is slower.
def count_models_by_enumeration2(solver, variables):
    solutions = 0
    for assignment in itertools.product([False, True], repeat=len(variables)):  # all combinations
        if solver.check([x if assign_true else Not(x) for x, assign_true in zip(variables, assignment)]) == sat:
            solutions = solutions + 1
    return solutions


# Using simplication instead of conditional checking is even slower.
def count_models_by_enumeration3(solver, variables):
    solutions = 0
    for assignment in itertools.product([BoolVal(False), BoolVal(True)], repeat=len(variables)):  # all combinations
        satisfied = True
        for assertion in solver.assertions():
            if is_false(simplify(substitute(assertion, list(zip(variables, assignment))))):
                satisfied = False
                break
        if satisfied: solutions = solutions + 1
    return solutions

def test_enu():
    x = Bools(' '.join('x' + str(i) for i in range(10)))
    solver = Solver()

    print('## OR formula ##')
    solver.add(Or(x))
    benchmark('Solver-based', count_models_with_solver, solver, x)
    benchmark('Enumeration-based (conditional check, direct assignment)', count_models_by_enumeration, solver, x)
    benchmark('Enumeration-based (conditional check, separate assignment)', count_models_by_enumeration2, solver, x)
    benchmark('Enumeration-based (substitute + simplify)', count_models_by_enumeration3, solver, x)

    print('\n## AND formula ##')
    solver.reset()
    solver.add(And(x))
    benchmark('Solver-based', count_models_with_solver, solver, x)
    benchmark('Enumeration-based (conditional check, direct assignment)', count_models_by_enumeration, solver, x)
    benchmark('Enumeration-based (conditional check, separate assignment)', count_models_by_enumeration2, solver, x)
    benchmark('Enumeration-based (substitute + simplify)', count_models_by_enumeration3, solver, x)


# test_enu()
