"""LLM-based abduction module.

This module provides tools for evaluating LLMs on abductive reasoning tasks
using SMT constraints as premises and conclusions.
"""

from arlib.llm.abduct.llm_abduct import (
    AbductionProblem,
    AbductionResult,
    LLMAbductor,
    AbductionEvaluator,
    get_variables,
    parse_smt2_string,
    extract_smt_from_llm_response
)

from arlib.llm.abduct.base import LLM, EnvLoader

# Try to import benchmark functions if available
try:
    from arlib.llm.abduct.benchmark import (
        generate_linear_arithmetic_problem,
        generate_boolean_problem,
        generate_mixed_problem,
        generate_benchmark_suite,
        save_benchmark,
        create_hardcoded_benchmarks
    )
    __has_benchmark = True
except ImportError:
    __has_benchmark = False

__all__ = [
    # Main classes
    "AbductionProblem",
    "AbductionResult",
    "LLMAbductor",
    "AbductionEvaluator",
    "LLM",
    "EnvLoader",
    
    # Utility functions
    "get_variables",
    "parse_smt2_string",
    "extract_smt_from_llm_response",
]

# Add benchmark functions if available
if __has_benchmark:
    __all__.extend([
        # Benchmark functions
        "generate_linear_arithmetic_problem",
        "generate_boolean_problem",
        "generate_mixed_problem",
        "generate_benchmark_suite",
        "save_benchmark",
        "create_hardcoded_benchmarks"
    ]) 