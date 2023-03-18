"""
A simple local solver implemented by ChatGPT...
TODO: examine its correctness
"""
import random
import time


def evaluate_clauses(clauses, variable_assignments):

    """
    Define a function to evaluate a set of clauses with a given set of variable assignments
    """
    for clause in clauses:
        clause_is_true = False
        for variable in clause:
            variable_value = variable_assignments[abs(variable)]
            if variable > 0 and variable_value or variable < 0 and not variable_value:
                clause_is_true = True
                break
        if not clause_is_true:
            return False
    return True


def local_search_solve_sat(clauses, max_iterations=10000):
    """
    Define the local search solver function
    """
    variable_assignments = {i: random.choice([True, False]) for i in range(1, max([abs(v) for clause in clauses for v in clause]) + 1)}
    for i in range(max_iterations):
        if evaluate_clauses(clauses, variable_assignments):
            return variable_assignments
        unsatisfied_clauses = [clause for clause in clauses if not evaluate_clauses([clause], variable_assignments)]
        variable_to_flip = random.choice(list(variable_assignments.keys()))
        variable_assignments[variable_to_flip] = not variable_assignments[variable_to_flip]
        if evaluate_clauses(unsatisfied_clauses, variable_assignments):
            continue
        else:
            variable_assignments[variable_to_flip] = not variable_assignments[variable_to_flip]
    return False


def test():
    example_clauses = [[1, 2, -3], [-1, -2, 3], [-1, 2, 3], [1, -2, -3], [-1, 2, -3]]
    start_time = time.time()
    local_search_solve_sat(example_clauses, 100)
    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))

test()