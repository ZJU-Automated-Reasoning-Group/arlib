"""
Parsing Abduction Inputs and Outputs.
We use the abduction extension of CVC5 as the syntax for input, and extract the problme as Z3py expression.

TODO: in many cases, we do not need this
"""

from typing import Tuple, Dict, Any, List
import re
import z3

from arlib.abduction.utils import extract_variables_from_smt2
from arlib.utils.sexpr import SExprParser


def balance_parentheses(expr: str) -> str:
    """
    Ensure that parentheses in the expression are balanced.
    
    Args:
        expr: Expression string
        
    Returns:
        Expression with balanced parentheses
    """
    # Count open and close parentheses
    open_count = expr.count('(')
    close_count = expr.count(')')
    
    # Add missing close parentheses if needed
    if open_count > close_count:
        expr = expr + ')' * (open_count - close_count)
    
    return expr


def extract_balanced_expr(text: str, start_idx: int = 0) -> str:
    """
    Extract a balanced expression with matching parentheses.
    
    Args:
        text: Input text
        start_idx: Starting index (default: 0)
        
    Returns:
        Balanced expression string
    """
    stack = []
    for i in range(start_idx, len(text)):
        if text[i] == '(':
            stack.append(i)
        elif text[i] == ')':
            if stack:
                stack.pop()
                if not stack:  # If stack is empty, we found a balanced expr
                    return text[start_idx:i+1]
    
    # If we get here, the expression is not balanced
    # Return the whole string and let balancing function handle it
    return text[start_idx:]


def get_sort_str(var: z3.ExprRef) -> str:
    """
    Get the SMT-LIB2 sort string for a Z3 variable.
    
    Args:
        var: Z3 variable
        
    Returns:
        Sort string in SMT-LIB2 format
    """
    # Handle different types of Z3 objects
    if hasattr(var, 'sort'):
        sort = var.sort()
        if z3.is_bv_sort(sort):
            # For bit-vectors, we need to specify the width
            bv_width = sort.size()
            return f"(_ BitVec {bv_width})"
        elif z3.is_array_sort(sort):
            # For arrays, format is (Array Domain Range)
            domain = str(sort.domain())
            range_sort = str(sort.range())
            return f"(Array {domain} {range_sort})"
        else:
            return str(sort)
    elif isinstance(var, z3.FuncDeclRef):
        # For function declarations, we need to handle them separately
        domain = []
        for i in range(var.arity()):
            domain.append(str(var.domain(i)))
        range_sort = str(var.range())
        return f"({' '.join(domain)}) {range_sort}"
    else:
        # Default for other types
        return "Int"


def parse_smt2_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """
    Parse a single SMT-LIB2 expression using Z3's parser.
    
    Args:
        expr_str: SMT-LIB2 expression string
        variables: Dictionary of variable names to Z3 variables
        
    Returns:
        Z3 expression
    """
    # Use the parse_expr function directly instead of z3.parse_smt2_string
    return parse_expr(expr_str, variables)


def extract_assertion(smt2_str: str, start: int) -> Tuple[str, int]:
    """
    Extract a complete assertion from the SMT-LIB2 string.
    
    Args:
        smt2_str: SMT-LIB2 string
        start: Starting index
        
    Returns:
        Tuple of (assertion string, next position)
    """
    # Find the start of the assertion body
    expr_start = smt2_str.find('(', start + 7)  # Skip '(assert '
    if expr_start == -1:
        expr_start = start + 7
    
    # Extract the balanced expression
    expr = extract_balanced_expr(smt2_str, expr_start)
    next_pos = smt2_str.find(')', expr_start + len(expr)) + 1
    
    return expr, next_pos


def extract_abduction_goal(smt2_str: str) -> str:
    """
    Extract the abduction goal from the SMT-LIB2 string.
    
    Args:
        smt2_str: SMT-LIB2 string
        
    Returns:
        Abduction goal string
    """
    # Find the get-abduct command
    idx = smt2_str.find('(get-abduct')
    if idx == -1:
        return None
    
    # Find the expression part (after the variable A)
    expr_start = smt2_str.find('(', idx + 10)  # Skip '(get-abduct'
    if expr_start == -1:
        return None
    
    # Extract the balanced expression
    expr = extract_balanced_expr(smt2_str, expr_start)
    return expr


def parse_abduction_problem(smt2_str: str) -> Tuple[z3.BoolRef, z3.BoolRef, Dict[str, Any]]:
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
    
    # Extract assertions (preconditions)
    assertions = []
    pos = smt2_str.find('(assert')
    while pos != -1:
        expr, next_pos = extract_assertion(smt2_str, pos)
        if expr:
            try:
                z3_expr = parse_expr(expr, variables)
                assertions.append(z3_expr)
            except Exception as e:
                print(f"Warning: Failed to parse assertion: {expr}")
                print(f"Error: {e}")
        pos = smt2_str.find('(assert', next_pos)
    
    # Extract abduction goal
    goal_expr = extract_abduction_goal(smt2_str)
    if not goal_expr:
        raise ValueError("No abduction goal found in the input")
    
    goal = parse_expr(goal_expr, variables)
    
    # Combine assertions into a single precondition
    if assertions:
        precond = z3.And(*assertions) if len(assertions) > 1 else assertions[0]
    else:
        precond = z3.BoolVal(True)  # No assertions means trivial precondition
    
    return precond, goal, variables


def parse_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """
    Parse an SMT-LIB2 expression to a Z3 expression.
    
    Args:
        expr_str: SMT-LIB2 expression string
        variables: Dictionary of variable names to Z3 variables
        
    Returns:
        Z3 expression
    """
    # Balance parentheses if needed
    balanced_expr = balance_parentheses(expr_str)
    
    # Use SExprParser to parse the expression into an S-expression
    try:
        s_expr = SExprParser.parse(balanced_expr)
        if s_expr is None:
            raise ValueError(f"Failed to parse expression: empty result")
        return _convert_sexpr_to_z3(s_expr, variables)
    except SExprParser.ParseError as e:
        # More specific error for parse errors
        raise ValueError(f"S-expression parse error: {str(e)}")
    except Exception as e:
        # General error handling
        raise ValueError(f"Failed to parse expression: {balanced_expr}. Error: {str(e)}")


def _convert_sexpr_to_z3(sexpr: Any, variables: Dict[str, Any]) -> z3.ExprRef:
    """
    Convert a parsed S-expression to a Z3 expression.
    
    Args:
        sexpr: Parsed S-expression
        variables: Dictionary of variable names to Z3 variables
        
    Returns:
        Z3 expression
    """
    # Handle atoms (variables, constants)
    if isinstance(sexpr, str):
        # Check if it's a variable
        if sexpr in variables:
            return variables[sexpr]
        # Check for boolean constants
        elif sexpr == "true":
            return z3.BoolVal(True)
        elif sexpr == "false":
            return z3.BoolVal(False)
        # Check for hexadecimal bit-vector literals - like #x00000064
        elif sexpr.startswith('#x'):
            # For now, assume all hex literals are for 32-bit bit-vectors
            # This is a simplification; in a real system, you'd infer the width
            value = int(sexpr[2:], 16)
            return z3.BitVecVal(value, 32)
        # Check for binary bit-vector literals - like #b1011
        elif sexpr.startswith('#b'):
            value = int(sexpr[2:], 2)
            width = len(sexpr) - 2
            return z3.BitVecVal(value, width)
        else:
            # Assume it's a symbol - use Int by default
            # This might need refinement based on context
            return z3.Int(sexpr)
    
    # Handle numbers
    elif isinstance(sexpr, (int, float)):
        return z3.IntVal(sexpr) if isinstance(sexpr, int) else z3.RealVal(sexpr)
    
    # Handle lists (applications of functions/operations)
    elif isinstance(sexpr, list) and sexpr:
        op = sexpr[0]
        args = [_convert_sexpr_to_z3(arg, variables) for arg in sexpr[1:]]
        
        # Handle arithmetic operations
        if op == "+":
            return sum(args[1:], args[0])
        elif op == "-":
            if len(args) == 1:
                return -args[0]
            else:
                return args[0] - sum(args[1:])
        elif op == "*":
            result = args[0]
            for arg in args[1:]:
                result = result * arg
            return result
        elif op == "div" or op == "/":
            return args[0] / args[1]
        
        # Handle comparison operations
        elif op == "=":
            return args[0] == args[1]
        elif op == "<":
            return args[0] < args[1]
        elif op == "<=":
            return args[0] <= args[1]
        elif op == ">":
            return args[0] > args[1]
        elif op == ">=":
            return args[0] >= args[1]
        
        # Handle boolean operations
        elif op == "and":
            return z3.And(*args)
        elif op == "or":
            return z3.Or(*args)
        elif op == "not":
            return z3.Not(args[0])
        elif op == "=>":
            return z3.Implies(args[0], args[1])
        
        # Handle bit-vector operations - standard ones
        elif op == "bvadd":
            return args[0] + args[1]
        elif op == "bvsub":
            return args[0] - args[1]
        elif op == "bvmul":
            return args[0] * args[1]
        elif op == "bvudiv":
            return z3.UDiv(args[0], args[1])
        elif op == "bvurem":
            return z3.URem(args[0], args[1])
        elif op == "bvslt":
            return args[0] < args[1]
        
        # Handle bit-vector comparison - these need special handling
        elif op == "bvult":
            return z3.ULT(args[0], args[1])
        elif op == "bvslt":
            return args[0] < args[1]
        elif op == "bvsle":
            return args[0] <= args[1]
        elif op == "bvule":
            return z3.ULE(args[0], args[1])
        elif op == "bvsgt":
            return args[0] > args[1]
        elif op == "bvugt":
            return z3.UGT(args[0], args[1])
        elif op == "bvsge":
            return args[0] >= args[1]
        elif op == "bvuge":
            return z3.UGE(args[0], args[1])
        
        # Array operations
        elif op == "select":
            return z3.Select(args[0], args[1])
        elif op == "store":
            return z3.Store(args[0], args[1], args[2])
        
        # If-Then-Else operation
        elif op == "ite":
            return z3.If(args[0], args[1], args[2])
        
        # Handle function applications
        elif op in variables and isinstance(variables[op], z3.FuncDeclRef):
            func = variables[op]
            return func(*args)
        
        else:
            # Try to interpret as a constant function application
            # This is useful for operations like "_" in bit-vector declarations
            if op == "_" and len(args) >= 2 and isinstance(args[0], z3.ExprRef):
                if str(args[0]) == "BitVec" and isinstance(args[1], z3.IntNumRef):
                    # Handle (_ BitVec N) pattern
                    width = args[1].as_long()
                    return z3.BitVec("result", width)
            
            raise ValueError(f"Unsupported operation: {op} in {sexpr}")
    
    else:
        raise ValueError(f"Unsupported S-expression: {sexpr}")


def example_int():
    """Example using integer variables"""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (declare-fun z () Int)
    (declare-fun w () Int)
    (declare-fun u () Int)
    (declare-fun v () Int)
    (assert (>= x 0))
    (assert (or (>= x 0) (< u v)))
    (get-abduct A (and (>= (+ x y z w u v) 2) (<= (+ x y z w) 3)))
    """
    
    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("== Integer Example ==")
        print("Variables:", list(vars.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


def example_bv():
    """Example using bit-vector variables"""
    smt2_str = """
    (declare-fun x () (_ BitVec 32))
    (declare-fun y () (_ BitVec 32))
    (declare-fun z () (_ BitVec 32))
    (assert (bvuge x #x00000000))
    (assert (bvult y #x00000064))
    (get-abduct A (bvuge (bvadd x y z) #x00000002))
    """
    
    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("\n== Bit-Vector Example ==")
        print("Variables:", list(vars.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


def example_mixed():
    """Example with mixed types"""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () (_ BitVec 32))
    (declare-fun arr () (Array Int Int))
    (declare-fun f (Int Int) Bool)
    (assert (>= x 0))
    (assert (bvult y #x00000064))
    (assert (= (select arr 5) 10))
    (get-abduct A (> x 5))
    """
    
    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("\n== Mixed Types Example ==")
        print("Variables:", list(vars.keys()))
        
        # Print variable types in a safer way
        for name, var in vars.items():
            if isinstance(var, z3.FuncDeclRef):
                arg_types = [str(var.domain(i)) for i in range(var.arity())]
                return_type = str(var.range())
                print(f"  {name}: Function({', '.join(arg_types)}) -> {return_type}")
            else:
                print(f"  {name}: {var.sort()}")
                
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_int()
    example_bv()
    example_mixed()

