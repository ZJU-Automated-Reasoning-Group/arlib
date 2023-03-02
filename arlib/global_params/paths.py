from pathlib import Path

project_root_dir = str(Path(__file__).parent.parent.parent)
z3_exec = project_root_dir + "/bin_solvers/z3"
cvc5_exec = project_root_dir + "/bin_solvers/cvc5"