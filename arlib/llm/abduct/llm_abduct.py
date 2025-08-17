"""
LLM-based abduction utilities (prompting + SMT validation).

This file now serves as a backward compatibility layer.
The implementation has been moved to separate modules for better organization.
"""

# Import everything from the new modular structure
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
from .utils import extract_smt_from_llm_response, parse_smt2_string

# Re-export all the main classes for backward compatibility
__all__ = [
    'AbductionProblem',
    'AbductionResult',
    'AbductionIterationResult',
    'FeedbackAbductionResult',
    'LLMAbductor',
    'FeedbackLLMAbductor',
    'validate_hypothesis',
    'generate_counterexample',
    'create_basic_prompt',
    'create_feedback_prompt',
    'extract_smt_from_llm_response',
    'parse_smt2_string'
]
