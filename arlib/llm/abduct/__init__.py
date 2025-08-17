"""LLM-based abduction module.

This module provides tools for evaluating LLMs on abductive reasoning tasks
using SMT constraints as premises and conclusions.
"""

# Import main classes and data structures
from .data_structures import (
    AbductionProblem,
    AbductionResult,
    AbductionIterationResult,
    FeedbackAbductionResult
)

from .base_abductor import LLMAbductor
from .feedback_abductor import FeedbackLLMAbductor
from .validation import validate_hypothesis, generate_counterexample
from .prompts import create_basic_prompt, create_feedback_prompt

# Import utilities
from .utils import extract_smt_from_llm_response, parse_smt2_string
from arlib.utils.z3_expr_utils import get_variables

# Keep backward compatibility by importing from llm_abduct
from .llm_abduct import LLMAbductor as LegacyLLMAbductor, FeedbackLLMAbductor as LegacyFeedbackLLMAbductor

"""Public API for LLM-based abduction."""

__all__ = [
    # Data structures
    'AbductionProblem',
    'AbductionResult',
    'AbductionIterationResult',
    'FeedbackAbductionResult',

    # Main classes
    'LLMAbductor',
    'FeedbackLLMAbductor',

    # Validation functions
    'validate_hypothesis',
    'generate_counterexample',

    # Prompt functions
    'create_basic_prompt',
    'create_feedback_prompt',

    # Utility functions
    'extract_smt_from_llm_response',
    'parse_smt2_string',
    'get_variables'
]
