"""
Sudoku solver using Z3
"""

from z3 import *
import sys, time

def create_sudoku_model():
    cells = [[Int(f"cell_{i}_{j}") for j in range(9)] for i in range(9)]
    cons = [And(1 <= cells[i][j], cells[i][j] <= 9) for i in range(9) for j in range(9)]
    cons += [Distinct(cells[i]) for i in range(9)]
    cons += [Distinct([cells[i][j] for i in range(9)]) for j in range(9)]
    cons += [Distinct([cells[3*I+i][3*J+j] for i in range(3) for j in range(3)]) for I in range(3) for J in range(3)]
    return cells, cons

def print_sudoku(m, cells):
    for i in range(9):
        print(("| " if i%3==0 else "  ") + " ".join(str(m.evaluate(cells[i][j])) + (" |" if j%3==2 else "") for j in range(9)))
        if i%3==2: print("+-------+-------+-------+")

def parse_puzzle(s):
    grid, row = [], []
    for c in s:
        if c in "123456789": row.append(int(c))
        elif c in "0.": row.append(0)
        if len(row)==9: grid.append(row); row=[]
    if row: grid.append(row)
    if len(grid)!=9: raise ValueError("Invalid puzzle format.")
    return grid

def solve_sudoku(puzzle):
    cells, cons = create_sudoku_model()
    cons += [cells[i][j]==puzzle[i][j] for i in range(9) for j in range(9) if puzzle[i][j]!=0]
    s = Solver(); s.add(cons)
    t0 = time.time(); res = s.check(); t1 = time.time()
    return (res==sat, s.model() if res==sat else None, cells, t1-t0)

example_puzzle = [
    [5,3,0,0,7,0,0,0,0], [6,0,0,1,9,5,0,0,0], [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3], [4,0,0,8,0,3,0,0,1], [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0], [0,0,0,4,1,9,0,0,5], [0,0,0,0,8,0,0,7,9]
]

def main():
    print("Z3 Sudoku Solver\n-----------------")
    if len(sys.argv)>1:
        with open(sys.argv[1]) as f:
            try: puzzle = parse_puzzle(f.read())
            except Exception as e: print(f"Error: {e}"); return
    else:
        puzzle = example_puzzle
        print("Using example puzzle:")
        for row in puzzle: print(" ".join(str(x) if x else "." for x in row))
    print("\nSolving...")
    solved, model, cells, t = solve_sudoku(puzzle)
    if solved:
        print(f"\nSolution found in {t:.4f} seconds:")
        print_sudoku(model, cells)
    else:
        print(f"\nNo solution exists (verified in {t:.4f} seconds)")

if __name__=="__main__": main()
