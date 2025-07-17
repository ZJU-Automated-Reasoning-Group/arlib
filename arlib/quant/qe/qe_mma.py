"""
Call Mathmatica for quantifier elimination
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


class MathematicaQE:
    """Interface to Mathematica for quantifier elimination."""
    def __init__(self, math_path: str = "math", timeout: int = 300):
        self.math_path = math_path
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str, domain: str = "Reals"):
        input_content = self._make_input(formula, domain)
        cmd = [self.math_path, "-noprompt", "-run"]
        cmd.append(f"<<{input_content}")
        success, stdout, stderr = run_external_tool(cmd, None, self.timeout)
        if not success:
            return False, stderr or stdout
        return True, self._parse_output(stdout)

    def _make_input(self, formula, domain):
        lines = [f"formula = {formula};",
                 f"result = Quiet[Resolve[formula, {domain}]];",
                 "Print[\"RESULT_START\"];",
                 "Print[result];",
                 "Print[\"RESULT_END\"];",
                 "Exit[];"]
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as f:
            f.write("\n".join(lines))
            return f.name

    def _parse_output(self, output):
        lines = output.strip().split('\n')
        result_lines = []
        capture = False
        for line in lines:
            if "RESULT_START" in line:
                capture = True
                continue
            elif "RESULT_END" in line:
                break
            elif capture:
                result_lines.append(line.strip())
        return ' '.join(result_lines)


def eliminate_quantifiers(formula: str, domain: str = "Reals",
                          math_path: Optional[str] = None,
                          timeout: int = 300) -> Tuple[bool, str]:
    """
    Convenience function to eliminate quantifiers from a formula using Mathematica.
    
    Args:
        formula: The formula with quantifiers to eliminate
        domain: The domain to use (Reals, Integers, etc.)
        math_path: Path to the Mathematica executable
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, result_formula)
    """
    qe = MathematicaQE(math_path, timeout)
    return qe.eliminate_quantifiers(formula, domain)


def smt2_to_mathematica(smt2_formula: str) -> str:
    """
    Convert an SMT-LIB2 formula to Mathematica syntax.
    This is a simplified version and may need to be extended for more complex formulas.
    
    Args:
        smt2_formula: Formula in SMT-LIB2 format
        
    Returns:
        Formula in Mathematica syntax
    """
    # This is a very basic conversion and would need to be expanded
    # for a real implementation
    formula = smt2_formula

    # Replace basic operators
    formula = formula.replace("(and ", "(")
    formula = formula.replace("(or ", "(")
    formula = formula.replace("(not ", "!")
    formula = formula.replace("(=> ", "Implies[")

    # Replace quantifiers
    formula = formula.replace("(forall ", "ForAll[")
    formula = formula.replace("(exists ", "Exists[")

    # Replace variables and constants
    formula = formula.replace("true", "True")
    formula = formula.replace("false", "False")

    return formula


def mathematica_to_smt2(math_formula: str) -> str:
    """
    Convert a Mathematica formula to SMT-LIB2 syntax.
    This is a simplified version and may need to be extended for more complex formulas.
    
    Args:
        math_formula: Formula in Mathematica syntax
        
    Returns:
        Formula in SMT-LIB2 format
    """
    # This is a very basic conversion and would need to be expanded
    # for a real implementation
    formula = math_formula

    # Replace basic operators
    formula = formula.replace("&&", " and ")
    formula = formula.replace("||", " or ")
    formula = formula.replace("!", "(not ")
    formula = formula.replace("Implies[", "(=> ")

    # Replace quantifiers
    formula = formula.replace("ForAll[", "(forall ")
    formula = formula.replace("Exists[", "(exists ")

    # Replace variables and constants
    formula = formula.replace("True", "true")
    formula = formula.replace("False", "false")

    return formula


def demo():
    """Demonstration of the Mathematica QE functionality."""
    # Example formula: ∃x. (x^2 - 3*x + 2 = 0)
    formula = "Exists[{x}, x^2 - 3*x + 2 == 0]"

    success, result = eliminate_quantifiers(formula)

    if success:
        print(f"Original formula: {formula}")
        print(f"After QE: {result}")
    else:
        print(f"QE failed: {result}")


if __name__ == "__main__":
    demo()
