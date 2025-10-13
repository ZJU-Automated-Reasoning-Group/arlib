"""Programming by Example (PBE) Solver.

This module provides a PBE solver that uses Version Space Algebra
to synthesize programs from input-output examples across multiple theories.
"""

from .pbe_solver import PBESolver
from .expression_generators import (
    generate_lia_expressions,
    generate_bv_expressions,
    generate_string_expressions
)

__all__ = ['PBESolver', 'generate_lia_expressions', 'generate_bv_expressions', 'generate_string_expressions']
