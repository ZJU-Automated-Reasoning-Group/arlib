"""
Call redlog for quantifier elimination
TODO: by LLM, to check
"""

import os
import subprocess
import tempfile
import time
import logging
from typing import List, Optional, Tuple, Union
from arlib.utils.misc import run_external_tool

logger = logging.getLogger(__name__)


class RedlogQE:
    """Interface to Redlog for quantifier elimination."""
    def __init__(self, redlog_path: str = "redlog", timeout: int = 300):
        self.redlog_path = redlog_path
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str, logic: str = "real"):
        input_content = self._make_input(formula, logic)
        cmd = [self.redlog_path, "-w"]
        success, stdout, stderr = run_external_tool(cmd, input_content, self.timeout)
        if not success:
            return False, stderr or stdout
        return True, self._parse_output(stdout)

    def _make_input(self, formula, logic):
        lines = ["load_package redlog;"]
        if logic.lower() == "real":
            lines.append("rlset ofsf;")
        elif logic.lower() == "integer":
            lines.append("rlset pasf;")
        else:
            lines.append(f"rlset {logic};")
        lines += [f"formula := {formula};", "result := rlqe formula;", "print_with result;", "quit;"]
        return "\n".join(lines)

    def _parse_output(self, output):
        return output.strip()


def eliminate_quantifiers(formula: str, logic: str = "real", redlog_path: str = None, timeout: int = 300):
    return RedlogQE(redlog_path or "redlog", timeout).eliminate_quantifiers(formula, logic)


def demo():
    """Demonstration of the Redlog QE functionality."""
    # Example formula: ∃x. (x^2 - 3*x + 2 = 0)
    formula = "ex({x}, x^2 - 3*x + 2 = 0)"

    success, result = eliminate_quantifiers(formula)

    if success:
        print(f"Original formula: {formula}")
        print(f"After QE: {result}")
    else:
        print(f"QE failed: {result}")


if __name__ == "__main__":
    demo()
