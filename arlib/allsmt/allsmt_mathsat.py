"""
An example of using MathSAT to enumerate all satisfying assignments.
(declare-fun x () Int)
(declare-fun y () Int)
(declare-fun a () Bool)
(declare-fun b () Bool)
(declare-fun c () Bool)
(declare-fun d () Bool)
(assert (= (> (+ x y) 0) a))
(assert (= (< (+ (* 2 x) (* 3 y)) (- 10)) c))
(assert (and (or a b) (or c d)))
;; enumerate all the consistent assignments (i.e. solutions) for the given
;; list of predicates. Notice that the arguments to check-allsat can only be
;; Boolean constants. If you need to enumerate over arbitrary theory atoms,
;; you can always "label" them with constants, as done above for
;; "(> (+ x y) 0)", labeled by "a"
(check-allsat (a b)
"""
import tempfile
import subprocess
import os
from z3 import *


def z3_to_smtlib2(solver, keys):
    """Convert Z3 constraints to SMT-LIB2 format with proper all-SAT tracking"""
    smtlib2 = "(set-logic QF_LIA)\n\n"

    # First declare all original variables
    for k in keys:
        smtlib2 += f"(declare-fun {k} () Int)\n"
    smtlib2 += "\n"

    # Create Boolean labels for tracking conditions
    bool_labels = []
    assertion_labels = {}
    label_counter = 0

    # Analyze assertions to create Boolean labels
    # FIXME: should only label assertions the user wants to track
    for assertion in solver.assertions():
        label_name = f"b{label_counter}"
        bool_labels.append(label_name)
        assertion_labels[label_name] = assertion
        smtlib2 += f"(declare-fun {label_name} () Bool)\n"
        label_counter += 1
    smtlib2 += "\n"

    # Add assertions with labels
    for label, assertion in assertion_labels.items():
        smtlib2 += f"(assert (= {label} {assertion.sexpr()}))\n"

    # Add original assertions
    for assertion in solver.assertions():
        smtlib2 += f"(assert {assertion.sexpr()})\n"
    smtlib2 += "\n"

    # Add check-allsat command with Boolean labels
    # FIXME: should only label assertions the user wants to track
    bool_labels_str = " ".join(bool_labels)
    smtlib2 += f"(check-allsat ({bool_labels_str}))\n"

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
            cmd = [mathsat_bin, tmp.name]

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
