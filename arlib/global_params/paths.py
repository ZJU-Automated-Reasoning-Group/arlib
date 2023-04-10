from pathlib import Path
import shutil

# list of solver names to check for executability
solver_names = {
    "z3": "z3",
    "cvc5": "cvc5",
    "yices2": "yices-smt2",
    "sharp_sat": "sharpSAT"
}

# FIXME: allow the user to specify the path to one ore more SMT solvers

# loop through the solvers and find their executable paths using shutil.which()
for name, exec_name in solver_names.items():
    # first, try to find the solvers in arlib/bin_solvers
    exec_path = str(Path(__file__).parent.parent.parent / "bin_solvers" / exec_name)
    if not shutil.which(exec_path):
        # in case the first step fails, try to find the solver in the system path.
        exec_path = shutil.which(exec_name)
    # update the corresponding variable with the found executable path
    globals()[f"{name}_exec"] = exec_path

# print the result for the last solver
# print(z3_exec)
# print(cvc5_exec)
# print(yices2_exec)
# print(sharp_sat_exec)

