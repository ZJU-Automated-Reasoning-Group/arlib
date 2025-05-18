"""
LLM provider factory for SMTO (Satisfiability Modulo Theories and Oracles)
This module provides a unified interface to various LLM providers for oracle handling.
"""

import logging
from typing import Optional, List, Dict, Any

from arlib.llm.smto.llm_providers import LLMConfig, LLMInterface

def create_llm(config: Optional[LLMConfig] = None) -> LLMInterface:
    """Factory function to create LLM interface based on config"""
    if config is None:
        config = LLMConfig()
    
    return LLMInterface(config)
