"""Z3 to CVC5 abduction interface."""
import z3
import tempfile
import subprocess
import os
from typing import Tuple


def z3_to_smtlib2_abduction(formula: z3.ExprRef, target_var: str) -> str:
    """Convert Z3 formula to SMT-LIB2 format with abduction syntax."""
    base_smt = formula.sexpr()
    logic = "QF_LIA" if contains_only_linear_arithmetic(formula) else "ALL"
    
    # Extract variable declarations
    var_decls = []
    for var_name, var_type in get_variables(formula).items():
        var_decls.append(f"(declare-fun {var_name} () {var_type})")

    return f"""(set-logic {logic})
{chr(10).join(var_decls)}
(assert true)
(get-abduct {target_var} {base_smt})
"""


def get_variables(formula: z3.ExprRef) -> dict:
    """Extract variable names and their types from a Z3 formula."""
    var_types = {}
    
    def collect_vars(f):
        if z3.is_const(f):
            var_name = str(f)
            if not (var_name == "True" or var_name == "False" or var_name.isdigit()):
                try:
                    int(var_name)  # Skip numerical constants
                except ValueError:
                    var_types[var_name] = f.sort()
        for child in f.children():
            collect_vars(child)
    
    collect_vars(formula)
    return var_types


def contains_only_linear_arithmetic(formula: z3.ExprRef) -> bool:
    """Check if formula contains only linear integer arithmetic."""
    has_non_lia = [False]
    
    def check(f):
        if z3.is_const(f) and not f.sort() == z3.IntSort():
            has_non_lia[0] = True
        elif z3.is_app(f):
            if f.decl().kind() in [z3.Z3_OP_MUL, z3.Z3_OP_DIV, z3.Z3_OP_IDIV, z3.Z3_OP_MOD]:
                children = f.children()
                if len(children) >= 2 and all(not z3.is_int_value(c) for c in children[:2]):
                    has_non_lia[0] = True
        for child in f.children():
            check(child)
    
    check(formula)
    return not has_non_lia[0]


def solve_abduction(formula: z3.ExprRef) -> Tuple[bool, str]:
    """Solve abduction problem using CVC5."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as tmp_file:
        smt2_content = z3_to_smtlib2_abduction(formula, "A")
        tmp_file.write(smt2_content)
        tmp_path = tmp_file.name
    
    try:
        # Get CVC5 path
        try:
            from arlib.global_params.paths import global_config
            cvc5_path = global_config.get_solver_path("cvc5")
        except (ImportError, AttributeError):
            cvc5_path = "cvc5"
            if subprocess.run(["which", "cvc5"], capture_output=True).returncode != 0:
                return False, "CVC5 not found in PATH"
        
        # Run CVC5
        cmd = [cvc5_path, '--produce-abducts', '--sygus-inference', tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Handle result
        if result.returncode != 0:
            return False, f"CVC5 error (code {result.returncode})"
        
        output = result.stdout.strip()
        if not output or 'unsat' in output.lower():
            return False, "No solution found"
        return True, output

    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass


# Example usage
if __name__ == "__main__":
    x = z3.Int('x')
    y = z3.Int('y')
    goal = z3.And(x > 0, y > x)
    
    print("Finding formula A such that A implies (x > 0 and y > x)")
    success, result = solve_abduction(goal)
    
    if success:
        print("Found abduction solution:", result)
    else:
        print("Failed:", result)
        