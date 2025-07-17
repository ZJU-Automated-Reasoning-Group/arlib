"""
Call qepcad for quantifier elimination
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


class QepcadQE:
    """Interface to QEPCAD for quantifier elimination."""
    def __init__(self, qepcad_path: str = "qepcad", timeout: int = 300):
        self.qepcad_path = qepcad_path
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str):
        input_content = self._make_input(formula)
        cmd = [self.qepcad_path, "+N500000000", "+L100000", "+H1000000"]
        success, stdout, stderr = run_external_tool(cmd, input_content, self.timeout)
        if not success:
            return False, stderr or stdout
        return True, self._parse_output(stdout)

    def _make_input(self, formula):
        qepcad_formula = self._convert_to_qepcad_syntax(formula)
        return f"[]\n{qepcad_formula}.\nfinish\n"

    def _parse_output(self, output):
        lines = output.strip().split('\n')
        result_lines = []
        capture = False
        for line in lines:
            if "An equivalent quantifier-free formula:" in line:
                capture = True
                continue
            elif capture and line.strip() and not line.startswith("="):
                result_lines.append(line.strip())
            elif capture and "=====================" in line:
                break
        return ' '.join(result_lines)

    def _convert_to_qepcad_syntax(self, formula: str) -> str:
        formula = formula.replace("∧", "/\\")
        formula = formula.replace("∨", "\\/")
        formula = formula.replace("¬", "~")
        formula = formula.replace("→", "==>")
        formula = formula.replace("∀", "A")
        formula = formula.replace("∃", "E")
        return formula


def eliminate_quantifiers(formula: str, qepcad_path: str = None, timeout: int = 300):
    return QepcadQE(qepcad_path or "qepcad", timeout).eliminate_quantifiers(formula)


def demo():
    """Demonstration of the QEPCAD QE functionality."""
    # Example formula: ∃x. (x^2 - 3*x + 2 = 0)
    formula = "(E x)[x^2 - 3*x + 2 = 0]"

    success, result = eliminate_quantifiers(formula)

    if success:
        print(f"Original formula: {formula}")
        print(f"After QE: {result}")
    else:
        print(f"QE failed: {result}")


if __name__ == "__main__":
    demo()
