# coding: utf-8
"""
Global configurations for arlib.
Manages solver paths, library dependencies, and system information.
"""
from typing import Dict
from pathlib import Path
from arlib.global_params import global_config

# SMT solver configurations
SMT_SOLVERS_PATH = {
    'z3': {
        'available': global_config.is_solver_available("z3"),
        'path': global_config.get_solver_path("z3"),
        'args': "-in"
    },
    'cvc5': {
        'available': global_config.is_solver_available("cvc5"),
        'path': global_config.get_solver_path("cvc5"),
        'args': "-q -i"
    },
    'mathsat': {
        'available': global_config.is_solver_available("mathsat"),
        'path': global_config.get_solver_path("mathsat"),
        'args': ""
    }
}

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
BIN_SOLVERS_PATH = PROJECT_ROOT / "bin_solvers"
BENCHMARKS_PATH = PROJECT_ROOT / "benchmarks"


if __name__ == "__main__":
    print(PROJECT_ROOT, BIN_SOLVERS_PATH, BENCHMARKS_PATH)
