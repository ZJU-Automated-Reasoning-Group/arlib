"""
Parallel SMT solving based on CDCL(T)

NOTE: Currently, the algorithm in this dir does not rely on arlib/arith, arlib/bv, arlib/fp, etc.
"""
from .smt_formula_manager import BooleanFormulaManager, TheoryFormulaManager, \
    SMTPreprocessor4Process
