"""
Parsing Abduction Inputs and Outputs
(Maybe we the abduction extension of CVC5 as the syntax for input, and extract the problme as Z3py expression?
"""

from typing import Tuple, Dict, Any
import re
import z3


def extract_variables_from_smt2(smt2_str: str) -> Dict[str, z3.ExprRef]:
    """
    Extract variables from SMT-LIB2 string.
    
    Args:
        smt2_str: SMT-LIB2 string with variable declarations
        
    Returns:
        Dictionary mapping variable names to Z3 variables
    """
    var_dict = {}
    
    # Extract variable declarations using regex
    var_decls = re.findall(r'\(declare-(?:fun|const)\s+(\w+)\s+\(\)\s+(\w+)\)', smt2_str)
    
    for var_name, var_type in var_decls:
        if var_type == 'Int':
            var_dict[var_name] = z3.Int(var_name)
        elif var_type == 'Real':
            var_dict[var_name] = z3.Real(var_name)
        elif var_type == 'Bool':
            var_dict[var_name] = z3.Bool(var_name)
    
    return var_dict


def parse_smt2_expr(expr_str: str, variables: Dict[str, z3.ExprRef]) -> z3.ExprRef:
    """
    Parse a single SMT-LIB2 expression using Z3's parser.
    
    Args:
        expr_str: SMT-LIB2 expression
        variables: Dictionary of variables
        
    Returns:
        Z3 expression
    """
    # Create SMT-LIB2 string with variable declarations
    decls = []
    for name, var in variables.items():
        sort = var.sort()
        sort_name = str(sort)
        decls.append(f"(declare-const {name} {sort_name})")
    
    smt2_str = "\n".join(decls + [f"(assert {expr_str})"])
    
    # Create a fresh solver and parse the expression
    s = z3.Solver()
    s.from_string(smt2_str)
    
    # Return the first assertion (our parsed expression)
    if s.assertions():
        return s.assertions()[0]
    return None


def parse_abduction_problem(smt2_str: str) -> Tuple[z3.BoolRef, z3.BoolRef, Dict[str, z3.ExprRef]]:
    """
    Parse an abduction problem from SMT-LIB2 format to Z3 formulas.
    
    Args:
        smt2_str: SMT-LIB2 string with abduction commands
        
    Returns:
        Tuple containing:
        - Z3 formula for the precondition (conjunction of assertions)
        - Z3 formula for the postcondition (abduction goal)
        - Dictionary of variables used in the formulas
    """
    # Extract variables
    variables = extract_variables_from_smt2(smt2_str)
    
    # Extract assertions
    assertions = []
    for match in re.finditer(r'\(assert\s+(.*?)\)', smt2_str, re.DOTALL):
        assertion_str = match.group(1)
        # Handle parentheses to make sure we have a complete expression
        # Count open and close parentheses
        open_count = assertion_str.count('(')
        close_count = assertion_str.count(')')
        
        # Add missing closing parentheses if needed
        if open_count > close_count:
            assertion_str += ')' * (open_count - close_count)
        
        expr = parse_smt2_expr(assertion_str, variables)
        if expr is not None:
            assertions.append(expr)
    
    # Extract abduction goal
    abduction_goal = None
    match = re.search(r'\(get-abduct\s+\w+\s+(.*)\)', smt2_str, re.DOTALL)
    if match:
        goal_str = match.group(1)
        # Handle parentheses
        # Find the balanced closing parenthesis
        paren_count = 0
        end_pos = len(goal_str)
        
        for i, char in enumerate(goal_str):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    end_pos = i
                    break
        
        goal_str = goal_str[:end_pos]
        
        # Add missing closing parentheses if needed
        open_count = goal_str.count('(')
        close_count = goal_str.count(')')
        if open_count > close_count:
            goal_str += ')' * (open_count - close_count)
        
        abduction_goal = parse_smt2_expr(goal_str, variables)
    
    # Combine assertions
    precondition = z3.And(*assertions) if assertions else z3.BoolVal(True)
    
    return precondition, abduction_goal, variables


def example():
    """Example usage of the parser"""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (declare-fun z () Int)
    (declare-fun w () Int)
    (declare-fun u () Int)
    (declare-fun v () Int)
    (assert (>= x 0))
    (get-abduct A (and (>= (+ x y z w u v) 2) (<= (+ x y z w) 3)))
    """
    
    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("Variables:", [str(v) for v in vars.values()])
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example()









