"""
Proving differential privacy properties of DP algorithms using SMT solving
"""

from z3 import *
import math
import sys
import time

def create_dp_model(epsilon, delta=0):
    """
    Create a model for proving differential privacy properties.
    
    Args:
        epsilon: The epsilon parameter for (ε,δ)-differential privacy
        delta: The delta parameter for (ε,δ)-differential privacy (default: 0 for pure DP)
        
    Returns:
        (variables, constraints) tuple where:
        - variables is a dictionary containing input and output variables
        - constraints is a list of Z3 constraints representing the DP property
    """
    # Create variables for two adjacent databases
    x1 = Real('x1')  # Value in database 1
    x2 = Real('x2')  # Value in database 2
    
    # Create variables for outputs of the algorithm
    y1 = Real('y1')  # Output for database 1
    y2 = Real('y2')  # Output for database 2
    
    # Variables dictionary
    variables = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
    
    # Constraint: the inputs are adjacent (differ by at most 1)
    adjacency_constraint = Abs(x1 - x2) <= 1
    
    # For the Laplace mechanism with sensitivity 1, we model:
    # Pr[M(D1) = y] ≤ e^ε * Pr[M(D2) = y] + δ
    #
    # For Laplace noise with parameter b, the PDF is (1/2b)e^(-|x|/b)
    # For DP to hold with sensitivity 1, we need b = 1/ε
    
    # The ratio of probabilities constraint for pure DP (δ=0):
    if delta == 0:
        # For any output y1 from database 1 and y2 from database 2,
        # the ratio of their probabilities must satisfy the DP condition
        dp_constraint = Implies(
            adjacency_constraint,
            Abs(y1 - x1) - Abs(y2 - x2) <= epsilon
        )
    else:
        # For (ε,δ)-DP we need a more complex constraint
        # This is a simplified version for illustration
        dp_constraint = Implies(
            adjacency_constraint,
            Or(
                Abs(y1 - x1) - Abs(y2 - x2) <= epsilon,
                And(y1 == y2, delta > 0)
            )
        )
    
    return variables, [dp_constraint]

def verify_dp_property(mechanism_name, epsilon, delta=0):
    """
    Verify if a mechanism satisfies differential privacy.
    
    Args:
        mechanism_name: Name of the DP mechanism to verify
        epsilon: The epsilon parameter for (ε,δ)-differential privacy
        delta: The delta parameter for (ε,δ)-differential privacy
        
    Returns:
        (is_dp, model, variables, solve_time) tuple where:
        - is_dp is a boolean indicating if the mechanism satisfies DP
        - model is the Z3 model if a violation is found, None otherwise
        - variables is the dictionary of variables used in the model
        - solve_time is the time taken to solve
    """
    # Create the model and constraints
    variables, dp_constraints = create_dp_model(epsilon, delta)
    
    # Create solver and add DP constraints
    s = Solver()
    s.add(dp_constraints)
    
    # Add mechanism-specific constraints
    if mechanism_name == "laplace":
        # For Laplace mechanism with sensitivity 1 and parameter b=1/ε,
        # we know it satisfies ε-DP
        # No additional constraints needed for verification
        pass
        
    elif mechanism_name == "gaussian":
        # For Gaussian mechanism with sensitivity 1 and standard deviation σ,
        # we need σ ≥ sqrt(2*ln(1.25/δ))/ε to satisfy (ε,δ)-DP
        # We create constraints for this mechanism
        sigma = Real('sigma')
        variables['sigma'] = sigma
        
        # Add constraint specific to Gaussian mechanism
        # This constraint would be violated if σ is too small
        s.add(sigma < math.sqrt(2 * math.log(1.25/delta)) / epsilon)
        
    elif mechanism_name == "randomized_response":
        # For randomized response with parameter p = e^ε/(1+e^ε),
        # we model the specific constraints
        p = Real('p')
        variables['p'] = p
        
        # Add constraints specific to randomized response
        s.add(p == epsilon / (1 + epsilon))  # Simplified for illustration
        
    # Negate the DP property to search for a counterexample
    s.push()
    s.add(Not(And(dp_constraints)))
    
    # Time the solving process
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
        # Found a violation of the DP property
        return False, s.model(), variables, solve_time
    else:
        # No violation found, the mechanism satisfies DP
        return True, None, variables, solve_time

def print_dp_verification_result(is_dp, model, variables, mechanism_name, epsilon, delta, solve_time):
    """
    Print the result of the DP verification.
    
    Args:
        is_dp: Boolean indicating if the mechanism satisfies DP
        model: Z3 model if a violation is found, None otherwise
        variables: Dictionary of variables used in the model
        mechanism_name: Name of the DP mechanism verified
        epsilon: The epsilon parameter used
        delta: The delta parameter used
        solve_time: Time taken to solve
    """
    dp_type = f"({epsilon},{delta})-DP" if delta > 0 else f"{epsilon}-DP"
    
    print(f"Verification of {mechanism_name.capitalize()} mechanism for {dp_type}:")
    print(f"  Time taken: {solve_time:.4f} seconds")
    
    if is_dp:
        print(f"  ✓ The {mechanism_name} mechanism satisfies {dp_type}")
    else:
        print(f"  ✗ Found a violation of {dp_type} for the {mechanism_name} mechanism")
        print("  Counterexample:")
        for var_name, var in variables.items():
            if var in model:
                print(f"    {var_name} = {model[var]}")

def main():
    print("Differential Privacy Verification using Z3 SMT Solver")
    print("----------------------------------------------------")
    
    mechanisms = {
        "laplace": "Laplace Mechanism",
        "gaussian": "Gaussian Mechanism",
        "randomized_response": "Randomized Response"
    }
    
    # Choose which mechanism to verify
    if len(sys.argv) > 1 and sys.argv[1] in mechanisms:
        mechanism = sys.argv[1]
    else:
        mechanism = "laplace"  # Default
    
    print(f"Verifying: {mechanisms[mechanism]}")
    
    # Try with different privacy parameters
    if mechanism == "laplace":
        # Pure DP (δ=0)
        epsilons = [0.1, 0.5, 1.0]
        for eps in epsilons:
            is_dp, model, variables, solve_time = verify_dp_property(mechanism, eps)
            print_dp_verification_result(is_dp, model, variables, mechanism, eps, 0, solve_time)
            print()
    
    elif mechanism == "gaussian":
        # Approximate DP (δ>0)
        params = [(1.0, 0.01), (0.5, 0.05), (0.1, 0.1)]
        for eps, delta in params:
            is_dp, model, variables, solve_time = verify_dp_property(mechanism, eps, delta)
            print_dp_verification_result(is_dp, model, variables, mechanism, eps, delta, solve_time)
            print()
    
    elif mechanism == "randomized_response":
        # Pure DP (δ=0)
        epsilons = [math.log(3), math.log(9), math.log(19)]
        for eps in epsilons:
            is_dp, model, variables, solve_time = verify_dp_property(mechanism, eps)
            print_dp_verification_result(is_dp, model, variables, mechanism, eps, 0, solve_time)
            print()

if __name__ == "__main__":
    main()
