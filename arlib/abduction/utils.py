import z3
from typing import Tuple, List


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