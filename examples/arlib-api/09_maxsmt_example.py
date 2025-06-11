#!/usr/bin/env python3
"""MaxSMT examples using arlib's optimization capabilities"""

import z3
from arlib.optimization.maxsmt import MaxSMTSolver, solve_maxsmt

def basic_maxsmt():
    print("=== Basic MaxSMT ===")
    x, y = z3.Ints('x y')
    hard = [x >= 0, y >= 0, x + y <= 10]
    soft = [x == 5, y == 5, x + y == 8]
    weights = [1.0, 1.0, 2.0]
    
    for alg in ["core-guided", "z3-opt"]:
        try:
            sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm=alg)
            if sat:
                print(f"{alg}: x={model.eval(x)}, y={model.eval(y)}, cost={cost}")
        except Exception as e:
            print(f"{alg}: Failed")

def scheduling_example():
    print("\n=== Scheduling ===")
    tasks = [z3.Bool(f"t{i}s{j}") for i in range(3) for j in range(2)]
    
    hard = []
    for i in range(3):
        task_vars = [tasks[i*2 + j] for j in range(2)]
        hard.append(z3.PbEq([(v, 1) for v in task_vars], 1))
    
    soft = [tasks[0], tasks[3], z3.Not(tasks[4])]
    weights = [2.0, 1.0, 1.0]
    
    try:
        solver = MaxSMTSolver("z3-opt")
        solver.add_hard_constraints(hard)
        solver.add_soft_constraints(soft, weights)
        sat, model, cost = solver.solve()
        
        if sat:
            for i in range(3):
                for j in range(2):
                    if z3.is_true(model.eval(tasks[i*2 + j])):
                        print(f"Task {i} -> Slot {j}")
            print(f"Cost: {cost}")
    except Exception as e:
        print(f"Failed: {e}")

def weighted_maxsmt():
    print("\n=== Weighted MaxSMT ===")
    x, y, z = z3.Ints('x y z')
    hard = [x >= 0, y >= 0, z >= 0, x + y + z <= 100]
    soft = [x >= 10, y >= 20, z >= 30, x + y >= 50]
    weights = [10.0, 5.0, 1.0, 5.0]
    
    try:
        sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm="z3-opt")
        if sat:
            print(f"x={model.eval(x)}, y={model.eval(y)}, z={model.eval(z)}, cost={cost}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    basic_maxsmt()
    scheduling_example()
    weighted_maxsmt() 