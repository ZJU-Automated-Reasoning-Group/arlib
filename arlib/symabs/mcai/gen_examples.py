import z3
import sys
import os
import argparse
from arlib.tests.formula_generator import FormulaGenerator
from arlib.utils.z3_expr_utils import get_variables

def main(tot: int, size: int, output_dir: str):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cnt = 0
    while cnt < tot:
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]
        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        var = get_variables(formula)
        if sol.check() == z3.sat and len(var) == len(variables):
            if cnt % size == 0 and not os.path.exists(f"{output_dir}/{cnt}_{cnt + size - 1}"):
                os.mkdir(f"{output_dir}/{cnt}_{cnt + size - 1}")
            with open(f"{output_dir}/{cnt - cnt % size}_{cnt - cnt % size + size - 1}/formula_{cnt}.smt2", "w") as f:
                f.write(sol.sexpr())
            cnt += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate examples for the MCAI project.")
    parser.add_argument(
        "-e", "--examples", 
        type=int, 
        help="Number of examples to generate.",
        default=200
    )
    parser.add_argument(
        "-s", "--size", 
        type=int, 
        help="Size of each batch of examples.",
        default=50
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        help="Output directory for the examples.",
        default="examples"
    )
    parser.add_argument(
        "-r", "--run",
        action="store_true",
        help="Run the examples after generation."
    )
    parser.add_argument(
        "-n", "--no_gen",
        action="store_true",
        help="Do not generate the examples."
    )
    args = parser.parse_args()
    cnt = args.examples
    size = args.size
    output_dir = args.output_dir
    try:
        if not args.no_gen:
            main(cnt, size, output_dir)
        if args.run:
            for i in range(0, cnt, size):
                os.system(f"python3 bv_mcai.py -d={output_dir}/{i}_{i + size - 1} -l=log/{i}_{i + size - 1}.log")
    except Exception as e:
        print(e)
        sys.exit(1)