# coding: utf-8
"""
Global configurations for arlib.
Manages solver paths, library dependencies, and system information.
"""
import os
import sys
import importlib
from typing import Dict
from pathlib import Path
from arlib.global_params import global_config
from arlib.utils.types import OSType


# SMT solver configurations
SMT_SOLVERS = {
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

# System information
if sys.platform == 'darwin':
    OS_TYPE = OSType.MAC
elif sys.platform == 'linux':
    OS_TYPE = OSType.LINUX
elif sys.platform == 'win32':
    OS_TYPE = OSType.WINDOWS
else:
    OS_TYPE = OSType.UNKNOWN

ARCH_TYPE = os.uname()[4]

# Required Python libraries and their availability
REQUIRED_LIBS = {
    'z3-solver': False,
    'numpy': False,
    'pyeda': False,
    'pysmt': False
}

def check_library(lib_name: str) -> bool:
    """Check if a Python library is available"""
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False

# Check library availability
for lib in REQUIRED_LIBS:
    REQUIRED_LIBS[lib] = check_library(lib.replace('-', '_'))

# Solver command strings with arguments
m_smt_solver_bin = f"{SMT_SOLVERS['z3']['path']} {SMT_SOLVERS['z3']['args']}" if SMT_SOLVERS['z3']['available'] else None
m_cvc5_solver_bin = f"{SMT_SOLVERS['cvc5']['path']} {SMT_SOLVERS['cvc5']['args']}" if SMT_SOLVERS['cvc5']['available'] else None

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIN_SOLVERS_PATH = PROJECT_ROOT / "bin_solvers"
BENCHMARKS_PATH = PROJECT_ROOT / "benchmarks"

def get_solver_status() -> Dict[str, bool]:
    """Get availability status of all solvers"""
    return {name: info['available'] for name, info in SMT_SOLVERS.items()}

def get_library_status() -> Dict[str, bool]:
    """Get availability status of all required libraries"""
    return REQUIRED_LIBS.copy()
