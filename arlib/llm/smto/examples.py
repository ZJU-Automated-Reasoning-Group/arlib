"""
Usage examples for SMTO solver.
"""

import z3
from arlib.llm.smto.smto import OraxSolver
from arlib.llm.smto.oracles import OracleInfo, WhiteboxOracleInfo, OracleAnalysisMode, OracleType


def blackbox_example():
    """Example usage of OraxSolver in blackbox mode"""
    # Initialize solver
    solver = OraxSolver(explanation_level="detailed")

    # Register an oracle for string length
    strlen_oracle = OracleInfo(
        name="strlen",
        input_types=[z3.StringSort()],
        output_type=z3.IntSort(),
        description="Calculate the length of a string",
        examples=[
            {"input": {"arg0": "hello"}, "output": "5"},
            {"input": {"arg0": "world!"}, "output": "6"}
        ]
    )
    solver.register_oracle(strlen_oracle)

    # Create variables and constraints
    s = z3.String('s')
    length = z3.Int('length')
    
    # Access the Z3 function declaration for strlen
    strlen = z3.Function('strlen', z3.StringSort(), z3.IntSort())

    # Add constraints
    solver.add_constraint(strlen(s) == length)
    solver.add_constraint(length > 5)
    solver.add_constraint(length < 10)

    # Solve
    model = solver.check()
    if model:
        print(f"Solution found: s = {model[s]}, length = {model[length]}")
        
        # Validate solution
        assert len(model[s].as_string()) == model[length].as_long()
    else:
        print("No solution found")
        
    # Print explanations
    for explanation in solver.get_explanations():
        print(f"{explanation['message']}")


def whitebox_example():
    """Example usage of OraxSolver in whitebox mode"""
    # Initialize solver with whitebox analysis enabled
    solver = OraxSolver(explanation_level="detailed", whitebox_analysis=True)

    # Sample source code for a function
    source_code = """
    def check_password(password):
        # Password must be at least 8 characters
        if len(password) < 8:
            return False
        
        # Password must contain at least one digit
        has_digit = False
        for char in password:
            if char.isdigit():
                has_digit = True
                break
        
        if not has_digit:
            return False
        
        # Password must contain at least one special character
        special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>/?"
        has_special = False
        for char in password:
            if char in special_chars:
                has_special = True
                break
                
        return has_special
    """
    
    # Register a whitebox oracle with source code
    password_oracle = WhiteboxOracleInfo(
        name="check_password",
        input_types=[z3.StringSort()],
        output_type=z3.BoolSort(),
        description="Check if a password meets security requirements",
        examples=[
            {"input": {"arg0": "password123"}, "output": "false"},
            {"input": {"arg0": "p@ssw0rd"}, "output": "true"}
        ],
        analysis_mode=OracleAnalysisMode.SOURCE_CODE,
        source_code=source_code
    )
    solver.register_oracle(password_oracle)

    # Create variables and constraints
    password = z3.String('password')
    
    # Access the Z3 function declaration
    check_pw = z3.Function('check_password', z3.StringSort(), z3.BoolSort())

    # Add constraints
    solver.add_constraint(check_pw(password) == True)
    
    # Add length constraint to limit search space
    solver.add_constraint(z3.Length(password) <= 12)

    # Solve
    model = solver.check()
    if model:
        print(f"Solution found: password = {model[password]}")
        
        # Print the symbolic model derived by the LLM
        print(f"Symbolic model: {solver.get_symbolic_model('check_password')}")
    else:
        print("No solution found")
        
    # Print explanations
    for explanation in solver.get_explanations():
        print(f"{explanation['message']}")


def custom_function_example():
    """Example of using a custom Python function as an oracle"""
    # Initialize solver
    solver = OraxSolver(explanation_level="basic")
    
    # Define a custom oracle function
    def is_prime(arg0):
        """Check if a number is prime"""
        if arg0 < 2:
            return False
        for i in range(2, int(arg0 ** 0.5) + 1):
            if arg0 % i == 0:
                return False
        return True
    
    # Register the function as an oracle
    prime_oracle = OracleInfo(
        name="is_prime",
        input_types=[z3.IntSort()],
        output_type=z3.BoolSort(),
        description="Check if a number is prime",
        examples=[
            {"input": {"arg0": 7}, "output": "true"},
            {"input": {"arg0": 10}, "output": "false"}
        ],
        oracle_type=OracleType.FUNCTION,
        function=is_prime
    )
    solver.register_oracle(prime_oracle)
    
    # Create variables and constraints
    x = z3.Int('x')
    is_prime_func = z3.Function('is_prime', z3.IntSort(), z3.BoolSort())
    
    # Find a prime number in a specific range
    solver.add_constraint(is_prime_func(x) == True)
    solver.add_constraint(x > 100)
    solver.add_constraint(x < 150)
    
    # Solve
    model = solver.check()
    if model:
        print(f"Found prime number: {model[x]}")
        # Validate
        assert is_prime(model[x].as_long())
    else:
        print("No solution found")


def documentation_analysis_example():
    """Example of using documentation for whitebox analysis"""
    # Initialize solver
    solver = OraxSolver(explanation_level="detailed", whitebox_analysis=True)
    
    # API documentation for a hypothetical encryption function
    documentation = """
    Function: encrypt_data(plaintext, key)
    
    Description:
    Encrypts plaintext data using a symmetric key.
    
    Parameters:
    - plaintext (string): The data to encrypt
    - key (string): The encryption key, must be exactly 16 characters
    
    Returns:
    - boolean: True if encryption was successful, False otherwise
    
    Behavior:
    1. If the key length is not exactly 16 characters, returns False
    2. If plaintext is empty, returns False
    3. Otherwise, performs encryption and returns True
    
    Examples:
    encrypt_data("secret data", "1234567890123456") -> True
    encrypt_data("data", "short_key") -> False
    """
    
    # Register the whitebox oracle
    encrypt_oracle = WhiteboxOracleInfo(
        name="encrypt_data",
        input_types=[z3.StringSort(), z3.StringSort()],
        output_type=z3.BoolSort(),
        description="Encrypt data with a symmetric key",
        examples=[
            {"input": {"arg0": "secret data", "arg1": "1234567890123456"}, "output": "true"},
            {"input": {"arg0": "data", "arg1": "short_key"}, "output": "false"}
        ],
        analysis_mode=OracleAnalysisMode.DOCUMENTATION,
        documentation=documentation
    )
    solver.register_oracle(encrypt_oracle)
    
    # Create variables and constraints
    plaintext = z3.String('plaintext')
    key = z3.String('key')
    encrypt_func = z3.Function('encrypt_data', z3.StringSort(), z3.StringSort(), z3.BoolSort())
    
    # Find valid encryption parameters
    solver.add_constraint(encrypt_func(plaintext, key) == True)
    
    # Solve
    model = solver.check()
    if model:
        print(f"Valid encryption parameters found:")
        print(f"Plaintext: {model[plaintext]}")
        print(f"Key: {model[key]}")
        print(f"Symbolic model: {solver.get_symbolic_model('encrypt_data')}")
    else:
        print("No solution found")


if __name__ == "__main__":
    print("=== Blackbox Example ===")
    blackbox_example()
    
    print("\n=== Whitebox Example ===")
    whitebox_example()
    
    print("\n=== Custom Function Example ===")
    custom_function_example()
    
    print("\n=== Documentation Analysis Example ===")
    documentation_analysis_example() 