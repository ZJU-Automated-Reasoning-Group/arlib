"""
MCP Solver integration for arlib.

This module provides Z3 solver integration through the MCP (Model Context Protocol) interface.
"""

from .z3 import Z3ModelManager, export_solution
from .prompts import load_prompt, get_prompt_path

# Server functionality (optional import)
try:
    from .server import create_server, serve
    __all__ = ["Z3ModelManager", "export_solution", "load_prompt", "get_prompt_path", "create_server", "serve"]
except ImportError:
    __all__ = ["Z3ModelManager", "export_solution", "load_prompt", "get_prompt_path"]
