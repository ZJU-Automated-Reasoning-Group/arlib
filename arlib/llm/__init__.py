"""
SMTO (Satisfiability Modulo Theories and Oracles) solver using LLM as oracle handler.

Supports two modes:
1. Blackbox mode: Traditional SMTO where we can only observe input-output behavior
2. Whitebox mode: Enhanced SMTO where we use LLM to analyze available component information
"""

# Import main solver
from arlib.llm.smto import OraxSolver

# Import oracle definitions
from arlib.llm.oracles import (
    OracleInfo,
    WhiteboxOracleInfo,
    OracleType,
    OracleAnalysisMode
)

# Import LLM configuration
from arlib.llm.llm_providers import LLMConfig

# Import utility classes
from arlib.llm.utils import OracleCache, ExplanationLogger

# Import example functions
from arlib.llm.examples import (
    blackbox_example,
    whitebox_example,
    custom_function_example,
    documentation_analysis_example
)

__all__ = [
    # Main solver
    "OraxSolver",
    
    # Oracle definitions
    "OracleInfo",
    "WhiteboxOracleInfo",
    "OracleType",
    "OracleAnalysisMode",
    
    # LLM configuration
    "LLMConfig",
    
    # Utility classes
    "OracleCache",
    "ExplanationLogger",
    
    # Example functions
    "blackbox_example",
    "whitebox_example",
    "custom_function_example",
    "documentation_analysis_example"
] 