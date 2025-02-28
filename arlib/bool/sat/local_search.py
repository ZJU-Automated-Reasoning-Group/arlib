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
    Define the local search solver function with improved variable selection
    """
    # Initialize random assignments
    variable_assignments = {i: random.choice([True, False]) for i in
                          range(1, max([abs(v) for clause in clauses for v in clause]) + 1)}
    
    best_unsat_count = len(clauses)
    stagnation_counter = 0
    
    for i in range(max_iterations):
        # Check if solution is found
        if evaluate_clauses(clauses, variable_assignments):
            print(f"Solution found at iteration {i}")
            return variable_assignments
            
        # Find unsatisfied clauses
        unsatisfied_clauses = [clause for clause in clauses if not evaluate_clauses([clause], variable_assignments)]
        current_unsat_count = len(unsatisfied_clauses)
        
        # Update best score and check for stagnation
        if current_unsat_count < best_unsat_count:
            best_unsat_count = current_unsat_count
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Early termination if stuck
        if stagnation_counter > 1000:
            print(f"Terminated early at iteration {i} due to stagnation")
            return False
            
        # Select variable from unsatisfied clause
        random_clause = random.choice(unsatisfied_clauses)
        variable_to_flip = abs(random.choice(random_clause))
        
        # Flip the variable
        variable_assignments[variable_to_flip] = not variable_assignments[variable_to_flip]
        
        # Keep the flip only if it improves unsatisfied clauses
        new_unsat_count = len([c for c in clauses if not evaluate_clauses([c], variable_assignments)])
        if new_unsat_count >= current_unsat_count:
            variable_assignments[variable_to_flip] = not variable_assignments[variable_to_flip]
    
    print(f"No solution found after {max_iterations} iterations")
    return False


def test():
    example_clauses = [[1, 2, -3], [-1, -2, 3], [-1, 2, 3], [1, -2, -3], [-1, 2, -3]]
    start_time = time.time()
    result = local_search_solve_sat(example_clauses, 1000)
    end_time = time.time()
    print("Time taken: {:.4f} seconds".format(end_time - start_time))
    if result:
        print("Solution found:", result)
        print("Verification:", evaluate_clauses(example_clauses, result))
    else:
        print("No solution found")


if __name__ == "__main__":
    test()
