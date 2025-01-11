import tempfile
import subprocess
import os
from z3 import *


def z3_to_smtlib2(solver, keys):
    """Convert Z3 constraints to SMT-LIB2 format"""
    # Get all assertions
    assertions = solver.assertions()

    # Start with setting logic and declaring variables
    smtlib2 = "(set-logic QF_LIA)\n"

    # Declare variables
    for k in keys:
        smtlib2 += f"(declare-fun {k} () Int)\n"

    # Add assertions
    for assertion in assertions:
        smtlib2 += f"(assert {assertion.sexpr()})\n"

    # Add check-sat
    smtlib2 += "(check-sat)\n"
    return smtlib2


def all_smt_mathsat(solver, keys):
    # Create temporary file for SMT-LIB2 instance
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as tmp:
        try:
            # Convert constraints to SMT-LIB2 and write to file
            smtlib2_content = z3_to_smtlib2(solver, keys)
            tmp.write(smtlib2_content)
            tmp.flush()

            # Prepare variable list for all-sat tracking
            var_list = " ".join(str(k) for k in keys)

            from arlib.global_params import global_config
            mathsat_bin = global_config.get_solver_path("mathsat")
            if not mathsat_bin:
                print("MathSAT not found")
                exit(1)
            # Call MathSAT with all-sat option
            cmd = [mathsat_bin, "-all-sat", "-all-sat-vars", var_list, tmp.name]

            # Run MathSAT and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate()

            # Parse and print solutions
            model_count = 0
            for line in output.splitlines():
                if line.startswith("sat"):
                    continue
                if line.startswith("unsat"):
                    break
                if line.strip():
                    model_count += 1
                    print(f"Model {model_count}:", line.strip())

        finally:
            # Clean up temporary file
            os.unlink(tmp.name)


def demo():
    x, y = Ints('x y')
    solver = Solver()
    solver.add(x + y == 5)
    solver.add(x > 0)
    solver.add(y > 0)
    all_smt_mathsat(solver, [x, y])


if __name__ == "__main__":
    demo()
