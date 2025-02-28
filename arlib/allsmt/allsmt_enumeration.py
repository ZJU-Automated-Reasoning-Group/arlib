from z3 import *

def all_smt(solver, keys) -> list:
    """
    Enumerate all models of the given SMT solver.
    """
    model_count = 0
    all_smt_models = []
    while solver.check() == sat:
        model = solver.model()
        model_count += 1
        # Print current model
        print([f"{k} = {model[k]}" for k in keys])
        all_smt_models.append(model)
        # Create blocking clause
        block = []
        for k in keys:
            block.append(k != model[k])
        solver.add(Or(block))
    return all_smt_models


def demo():
    x, y = Ints('x y')
    solver = Solver()
    solver.add(x + y == 5)
    solver.add(x > 0)
    solver.add(y > 0)
    all_smt(solver, [x, y])

if __name__ == "__main__":
    demo()