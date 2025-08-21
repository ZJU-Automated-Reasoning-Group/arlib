"""
Template functions for Z3 quantifiers and common constraint patterns.
"""

# Import and expose Z3 templates
from .z3_templates import (
    all_distinct,
    array_contains,
    array_is_sorted,
    at_least_k,
    at_most_k,
    exactly_k,
    function_is_injective,
    function_is_surjective,
)

# Import and expose function templates
from .function_templates import (
    array_template,
    constraint_satisfaction_template,
    demo_template,
    optimization_template,
    quantifier_template,
)

# Import and expose subset templates
from .subset_templates import smallest_subset_with_property

__all__ = [
    # Z3 templates
    "array_is_sorted",
    "all_distinct",
    "array_contains",
    "exactly_k",
    "at_most_k",
    "at_least_k",
    "function_is_injective",
    "function_is_surjective",
    # Function templates
    "constraint_satisfaction_template",
    "optimization_template",
    "array_template",
    "quantifier_template",
    "demo_template",
    # Subset templates
    "smallest_subset_with_property",
]
