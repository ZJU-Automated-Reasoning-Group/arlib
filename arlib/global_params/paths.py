from pathlib import Path

# FIXME: 1. try to automatically find the binary solvers in the current system (e.g., /usr/lib/bin..
#        2. allow the user to specify the path to one ore more SMT solvers
project_root_dir = str(Path(__file__).parent.parent.parent)
z3_exec = project_root_dir + "/bin_solvers/z3"
cvc5_exec = project_root_dir + "/bin_solvers/cvc5"