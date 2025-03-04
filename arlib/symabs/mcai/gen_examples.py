import z3
from arlib.tests.formula_generator import FormulaGenerator
from arlib.utils.z3_expr_utils import get_variables

def main():
    cnt = 0
    while cnt < 100:
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]
        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        var = get_variables(formula)
        if sol.check() == z3.sat and len(var) == len(variables):
            with open(f"examples/formula_{cnt}.smt2", "w") as f:
                f.write(sol.sexpr())
            cnt += 1

if __name__ == '__main__':
    main()