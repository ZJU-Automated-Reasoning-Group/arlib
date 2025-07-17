"""
Graph coloring using SMT solving
"""

from z3 import *
import sys, time

def create_graph_coloring_model(graph, num_colors):
    colors = {v: Int(f"color_{v}") for v in graph}
    constraints = [And(0 <= colors[v], colors[v] < num_colors) for v in graph]
    constraints += [colors[v] != colors[n] for v in graph for n in graph[v] if v < n]
    return colors, constraints

def solve_graph_coloring(graph, num_colors):
    colors, constraints = create_graph_coloring_model(graph, num_colors)
    s = Solver(); s.add(constraints)
    t0 = time.time(); result = s.check(); t1 = time.time()
    return (result == sat, s.model() if result == sat else None, colors if result == sat else None, t1 - t0)

def print_coloring(model, colors):
    coloring = {}
    for v in colors:
        c = model.evaluate(colors[v]).as_long()
        coloring.setdefault(c, []).append(v)
    for color in sorted(coloring):
        print(f"  Color {color}: {', '.join(map(str, sorted(coloring[color])))}")

def main():
    cycle5 = {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]}
    petersen = {0: [1, 4, 5], 1: [0, 2, 6], 2: [1, 3, 7], 3: [2, 4, 8], 4: [0, 3, 9], 5: [0, 7, 8], 6: [1, 8, 9], 7: [2, 5, 9], 8: [3, 5, 6], 9: [4, 6, 7]}
    k4 = {i: [j for j in range(4) if j != i] for i in range(4)}
    examples = {"cycle5": (cycle5, "5-Cycle"), "petersen": (petersen, "Petersen Graph"), "k4": (k4, "Complete Graph K4")}
    graph, name = examples.get(sys.argv[1], examples["petersen"]) if len(sys.argv) > 1 else examples["petersen"]
    print(f"Graph Coloring using Z3 SMT Solver\n----------------------------------\nUsing {name}")
    for num_colors in range(2, 5):
        print(f"\nAttempting coloring with {num_colors} colors:")
        colorable, model, colors, t = solve_graph_coloring(graph, num_colors)
        if colorable:
            print(f"Solution found in {t:.4f} seconds"); print_coloring(model, colors)
        else:
            print(f"No valid coloring exists with {num_colors} colors (verified in {t:.4f} seconds)")

if __name__ == "__main__":
    main()
