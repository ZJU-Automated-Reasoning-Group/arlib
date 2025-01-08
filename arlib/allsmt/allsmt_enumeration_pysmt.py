from pysmt.shortcuts import *

def all_smt(solver, keys):
    model_count = 0
    while solver.solve():
        model = solver.get_model()
        model_count += 1
        # Print current model
        print([f"{k} = {model[k]}" for k in keys])
        # Create blocking clause
        block = []
        for k in keys:
            block.append(Not(Equals(k, model[k])))
        solver.add_assertion(Or(block))

def demo():
    x = Symbol('x', INT)
    y = Symbol('y', INT)
    
    solver = Solver()
    solver.add_assertion(Equals(Plus(x, y), Int(5)))
    solver.add_assertion(GT(x, Int(0)))
    solver.add_assertion(GT(y, Int(0)))
    
    all_smt(solver, [x, y])

if __name__ == "__main__":
    demo()
    