from pysmt.shortcuts import *


def all_smt(expr, keys) -> list:
    """
    Enumerate all models
    """
    solver = Solver()
    solver.add_assertion(expr)
    model_count = 0
    all_smt_models = []
    while solver.solve():
        model = solver.get_model()
        model_count += 1
        # Print current model
        print([f"{k} = {model[k]}" for k in keys])
        all_smt_models.append(model)
        # Create blocking clause
        block = []
        for k in keys:
            block.append(Not(Equals(k, model[k])))
        solver.add_assertion(Or(block))
    print(f"Total number of models: {model_count}")
    return all_smt_models


def demo():
    x = Symbol('x', INT)
    y = Symbol('y', INT)

    all_smt(And(Equals(Plus(x, y), Int(5)), GT(x, Int(0)), GT(y, Int(0))), 
            [x, y])


if __name__ == "__main__":
    demo()
    