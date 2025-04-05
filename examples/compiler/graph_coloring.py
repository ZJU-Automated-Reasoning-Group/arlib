"""
Graph coloring using SMT solving
"""

from z3 import *
import sys
import time

def create_graph_coloring_model(graph, num_colors):
    """
    Create variables and constraints for the graph coloring problem.
    
    Args:
        graph: Dictionary where keys are vertices and values are lists of adjacent vertices
        num_colors: Number of colors to use
        
    Returns:
        (variables, constraints) tuple where:
        - variables is a dictionary mapping each vertex to its color variable
        - constraints is a list of Z3 constraints
    """
    # Create color variables for each vertex (colors are integers from 0 to num_colors-1)
    colors = {v: Int(f"color_{v}") for v in graph}
    
    # Each vertex must have a color between 0 and num_colors-1
    domain_constraints = [And(0 <= colors[v], colors[v] < num_colors) for v in graph]
    
    # Adjacent vertices must have different colors
    edge_constraints = []
    for v in graph:
        for neighbor in graph[v]:
            if v < neighbor:  # Only add constraint once per edge
                edge_constraints.append(colors[v] != colors[neighbor])
    
    return colors, domain_constraints + edge_constraints

def solve_graph_coloring(graph, num_colors):
    """
    Solve the graph coloring problem.
    
    Args:
        graph: Dictionary where keys are vertices and values are lists of adjacent vertices
        num_colors: Number of colors to use
        
    Returns:
        (is_colorable, model, colors, solve_time) tuple where:
        - is_colorable is a boolean indicating if the graph can be colored with num_colors
        - model is the Z3 model if colorable, None otherwise
        - colors is the dictionary of color variables
        - solve_time is the time taken to solve
    """
    # Create the model and constraints
    colors, constraints = create_graph_coloring_model(graph, num_colors)
    
    # Create solver and add all constraints
    s = Solver()
    s.add(constraints)
    
    # Time the solving process
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
        return True, s.model(), colors, solve_time
    else:
        return False, None, None, solve_time

def print_coloring(model, colors):
    """
    Print the coloring solution.
    
    Args:
        model: Z3 model containing the solution
        colors: Dictionary mapping vertices to color variables
    """
    # Group vertices by color
    coloring = {}
    for v in colors:
        color_val = model.evaluate(colors[v]).as_long()
        if color_val not in coloring:
            coloring[color_val] = []
        coloring[color_val].append(v)
    
    print("Graph coloring:")
    for color in sorted(coloring.keys()):
        vertices = coloring[color]
        print(f"  Color {color}: {', '.join(sorted(str(v) for v in vertices))}")

def main():
    # Example 1: Simple graph (a cycle with 5 vertices)
    cycle5 = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 0]
    }
    
    # Example 2: Petersen graph (requires at least 3 colors)
    petersen = {
        0: [1, 4, 5],
        1: [0, 2, 6],
        2: [1, 3, 7],
        3: [2, 4, 8],
        4: [0, 3, 9],
        5: [0, 7, 8],
        6: [1, 8, 9],
        7: [2, 5, 9],
        8: [3, 5, 6],
        9: [4, 6, 7]
    }
    
    # Example 3: Complete graph K4 (requires 4 colors)
    k4 = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2]
    }
    
    print("Graph Coloring using Z3 SMT Solver")
    print("----------------------------------")
    
    # Choose which graph to use
    examples = {
        "cycle5": (cycle5, "5-Cycle"),
        "petersen": (petersen, "Petersen Graph"),
        "k4": (k4, "Complete Graph K4")
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        graph, name = examples[sys.argv[1]]
    else:
        graph, name = examples["petersen"]  # Default
    
    print(f"Using {name}")
    
    # Try with different numbers of colors
    for num_colors in range(2, 5):
        print(f"\nAttempting coloring with {num_colors} colors:")
        
        colorable, model, colors, solve_time = solve_graph_coloring(graph, num_colors)
        
        if colorable:
            print(f"Solution found in {solve_time:.4f} seconds")
            print_coloring(model, colors)
        else:
            print(f"No valid coloring exists with {num_colors} colors (verified in {solve_time:.4f} seconds)")

if __name__ == "__main__":
    main()
