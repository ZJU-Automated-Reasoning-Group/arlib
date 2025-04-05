"""
Solving non-linear formulas via incremental linearization
"""

from z3 import *


def abstract_to_linear(formula):
    # Abstract non-linear parts to uninterpreted functions
    abstracted_formula = formula
    for expr in post_order_traverse(formula):
        if is_non_linear(expr):
            # Create uninterpreted function with same domain and range
            uf_name = f"nl_{expr.decl().name()}"
            domain = [arg.sort() for arg in expr.children()]
            uf = Function(uf_name, *domain, expr.sort())
            # Replace non-linear expression with UF application
            abstracted_formula = substitute(abstracted_formula,
                                            (expr, uf(*expr.children())))
    return abstracted_formula


def is_non_linear(expr):
    """Check if an expression is non-linear"""
    if not is_app(expr):
        return False
    # Non-linear multiplication
    if expr.decl().kind() == Z3_OP_MUL:
        args = expr.children()
        num_vars = sum(1 for arg in args if is_var(arg))
        return num_vars > 1
    # Transcendental functions are non-linear
    op = expr.decl().kind()
    return op in [Z3_OP_UNINTERPRETED] or str(expr.decl()) in ['sin', 'cos', 'tan', 'exp', 'log']


def is_var(expr):
    """Check if expression is a variable"""
    return is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED


def post_order_traverse(formula):
    """Traverse formula in post-order"""
    visited = set()
    result = []

    def traverse(expr):
        if expr in visited:
            return
        if is_app(expr):
            for child in expr.children():
                traverse(child)
        visited.add(expr)
        result.append(expr)

    traverse(formula)
    return result


def is_unsat(formula):
    solver = Solver()
    solver.add(formula)
    return solver.check() == unsat


def validate_model(formula, model):
    """Validate if a model satisfies the formula in the non-linear world"""
    # Substitute model values into the formula
    substituted = substitute_model(formula, model)
    # Check if the substituted formula is valid
    solver = Solver()
    solver.add(Not(substituted))
    return solver.check() == unsat


def substitute_model(formula, model):
    """Substitute model values into a formula"""
    substitutions = []
    for decl in model.decls():
        val = model[decl]
        if is_algebraic_value(val):
            # Handle algebraic numbers
            val = val.approx(20)  # Use 20 digits of precision
        if decl.arity() == 0:  # Only substitute constants/variables
            substitutions.append((decl(), val))
    return substitute(formula, substitutions)


def generate_refinement(formula, model):
    """Generate refinement constraints based on the current model
    TODO: also add refinement based on axioms of non-linear parts (e.g., multiplication, division, exponentiation, logarithm, trigonometry, etc.)
    """
    refinements = []
    for expr in post_order_traverse(formula):
        if is_non_linear(expr):
            # Get the UF that replaced this expression
            uf_name = f"nl_{expr.decl().name()}"
            uf = Function(uf_name, *[arg.sort() for arg in expr.children()], expr.sort())
            uf_app = uf(*expr.children())

            # Add refinement based on concrete values
            concrete_val = substitute_model(expr, model)
            refinements.append(uf_app == concrete_val)

            # Add additional properties (e.g., monotonicity for multiplication)
            if expr.decl().kind() == Z3_OP_MUL:
                x, y = expr.children()
                if is_var(x) and is_var(y):
                    # Monotonicity: x1 ≤ x2 ∧ y ≥ 0 → mul(x1,y) ≤ mul(x2,y)
                    x1, x2 = Consts('x1 x2', x.sort())
                    refinements.append(ForAll([x1, x2, y],
                                              Implies(And(x1 <= x2, y >= 0),
                                                      uf(x1, y) <= uf(x2, y))))

    return And(refinements)


def incremental_linearization(non_linear_formula):
    """Solve non-linear formula using incremental linearization"""
    # Initial abstraction
    linear_formula = abstract_to_linear(non_linear_formula)
    solver = Solver()
    solver.add(linear_formula)

    while True:
        if solver.check() == unsat:
            return "UNSAT"

        model = solver.model()
        if validate_model(non_linear_formula, model):
            return "SAT", model

        # Model is spurious, refine abstraction
        refinement = generate_refinement(non_linear_formula, model)
        solver.add(refinement)


if __name__ == "__main__":
    x, y = Reals('x y')
    non_linear_formula = And(x * y > 3, x + y < 5)  # Placeholder non-linear formula
    result = incremental_linearization(non_linear_formula)
    print(result)
