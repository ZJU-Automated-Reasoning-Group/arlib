"""
Sudoku solver using Z3
"""

from z3 import *
import sys
import time

def create_sudoku_model():
    """
    Create a 9x9 grid of integer variables for Sudoku.
    Each cell is an integer between 1 and 9.
    """
    # Create 9x9 matrix of integer variables
    cells = [[Int(f"cell_{i}_{j}") for j in range(9)] for i in range(9)]
    
    # Each cell contains a value in {1, ..., 9}
    cells_c = [And(1 <= cells[i][j], cells[i][j] <= 9) 
               for i in range(9) for j in range(9)]
    
    # Each row contains distinct values
    rows_c = [Distinct(cells[i]) for i in range(9)]
    
    # Each column contains distinct values
    cols_c = [Distinct([cells[i][j] for i in range(9)]) 
              for j in range(9)]
    
    # Each 3x3 square contains distinct values
    squares_c = [Distinct([cells[3*I+i][3*J+j] 
                          for i in range(3) for j in range(3)]) 
                for I in range(3) for J in range(3)]
    
    # Combine all constraints
    sudoku_c = cells_c + rows_c + cols_c + squares_c
    
    return cells, sudoku_c

def print_sudoku(model, cells):
    """
    Print the Sudoku solution from the model.
    """
    print("+-------+-------+-------+")
    for i in range(9):
        print("|", end=" ")
        for j in range(9):
            if j % 3 == 2:
                print(f"{model.evaluate(cells[i][j])}", end=" |")
            else:
                print(f"{model.evaluate(cells[i][j])}", end=" ")
        print()
        if i % 3 == 2:
            print("+-------+-------+-------+")

def add_puzzle_constraints(cells, puzzle):
    """
    Add constraints for initial puzzle values.
    puzzle: 9x9 matrix, 0 represents empty cells
    """
    constraints = []
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                constraints.append(cells[i][j] == puzzle[i][j])
    return constraints

def solve_sudoku(puzzle):
    """
    Solve a Sudoku puzzle.
    puzzle: 9x9 matrix, 0 represents empty cells
    Returns: (is_solved, model, solve_time)
    """
    # Create model and constraints
    cells, constraints = create_sudoku_model()
    
    # Add puzzle constraints
    puzzle_c = add_puzzle_constraints(cells, puzzle)
    
    # Create solver and add all constraints
    s = Solver()
    s.add(constraints + puzzle_c)
    
    # Time the solving process
    start_time = time.time()
    result = s.check()
    solve_time = time.time() - start_time
    
    if result == sat:
        return True, s.model(), cells, solve_time
    else:
        return False, None, None, solve_time

def parse_puzzle(puzzle_str):
    """
    Parse a string representation of a puzzle.
    Accepts formats with spaces, newlines, and characters '.', '0', or any digit.
    """
    grid = []
    row = []
    
    for c in puzzle_str:
        if c in "123456789":
            row.append(int(c))
            if len(row) == 9:
                grid.append(row)
                row = []
        elif c in "0.":
            row.append(0)  # Empty cell
            if len(row) == 9:
                grid.append(row)
                row = []
    
    # Handle case where the last row might not have been added
    if row and len(row) == 9:
        grid.append(row)
        
    # Ensure we have 9 rows
    if len(grid) != 9:
        raise ValueError("Invalid puzzle format. Expected 9 rows.")
    
    return grid

# Example puzzle (0 represents empty cells)
# This is a hard puzzle from https://en.wikipedia.org/wiki/Sudoku
example_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Another example (empty puzzle for user input)
empty_puzzle = [[0 for _ in range(9)] for _ in range(9)]

def main():
    print("Z3 Sudoku Solver")
    print("-----------------")
    
    # Decide which puzzle to solve
    if len(sys.argv) > 1:
        # If a filename is provided, read the puzzle from file
        with open(sys.argv[1], 'r') as f:
            puzzle_str = f.read()
            try:
                puzzle = parse_puzzle(puzzle_str)
            except ValueError as e:
                print(f"Error: {e}")
                return
    else:
        # Otherwise use the example puzzle
        puzzle = example_puzzle
        print("Using example puzzle:")
        for row in puzzle:
            print(" ".join(str(x) if x != 0 else "." for x in row))
    
    print("\nSolving...")
    solved, model, cells, solve_time = solve_sudoku(puzzle)
    
    if solved:
        print(f"\nSolution found in {solve_time:.4f} seconds:")
        print_sudoku(model, cells)
    else:
        print(f"\nNo solution exists (verified in {solve_time:.4f} seconds)")

if __name__ == "__main__":
    main()
