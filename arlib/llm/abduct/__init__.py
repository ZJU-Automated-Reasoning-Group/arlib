"""LLM-based abduction module.

This module provides tools for evaluating LLMs on abductive reasoning tasks
using SMT constraints as premises and conclusions.
"""

from arlib.llm.abduct.llm_abduct import (
    AbductionProblem,
    AbductionResult,
    LLMAbductor,
)
from arlib.llm.abduct.utils import (
    parse_smt2_string,
    extract_smt_from_llm_response,
)
from arlib.utils.z3_expr_utils import get_variables

from arlib.llm.abduct.evaluator import AbductionEvaluator

from arlib.llm.abduct.base import LLM, EnvLoader, LLMViaTool


__all__ = [
    # Main classes
    "AbductionProblem",
    "AbductionResult",
    "LLMAbductor",
    "AbductionEvaluator",
    "LLM",
    "EnvLoader",
    "LLMViaTool",

    # Utility functions
    "get_variables",
    "parse_smt2_string",
    "extract_smt_from_llm_response",
]
