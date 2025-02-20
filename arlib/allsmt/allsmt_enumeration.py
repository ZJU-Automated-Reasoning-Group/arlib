from z3 import *

def all_smt(solver, keys):
    """
    Enumerate all models of the given SMT solver.
    """
    model_count = 0
    while solver.check() == sat:
        model = solver.model()
        model_count += 1
        # Print current model
        print([f"{k} = {model[k]}" for k in keys])
        # Create blocking clause
        block = []
        for k in keys:
            block.append(k != model[k])
        solver.add(Or(block))


def demo():
    x, y = Ints('x y')
    solver = Solver()
    solver.add(x + y == 5)
    solver.add(x > 0)
    solver.add(y > 0)
    all_smt(solver, [x, y])

if __name__ == "__main__":
    demo()