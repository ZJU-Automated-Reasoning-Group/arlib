"""Version Space Algebra for Program Synthesis.

This module provides an algebraic framework for representing and manipulating
sets of programs that are consistent with observed input-output examples.
"""

from .vsa import VersionSpace, VSAlgebra
from .expressions import (
    Expression, Variable, Constant, BinaryOp, UnaryOp,
    IfExpr, LoopExpr, FunctionCallExpr, Theory
)

__all__ = [
    'VersionSpace', 'VSAlgebra', 'Expression', 'Variable', 'Constant',
    'BinaryOp', 'UnaryOp', 'IfExpr', 'LoopExpr', 'FunctionCallExpr', 'Theory'
]
