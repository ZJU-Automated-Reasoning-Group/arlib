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

logger = logging.getLogger(__name__)


class MathematicaQE:
    """Interface to Mathematica for quantifier elimination."""

    def __init__(self, math_path: Optional[str] = None, timeout: int = 300):
        """
        Initialize the Mathematica QE interface.
        
        Args:
            math_path: Path to the Mathematica executable (math or MathKernel).
                      If None, assumes 'math' is in PATH.
            timeout: Timeout in seconds for Mathematica execution.
        """
        self.math_path = math_path if math_path else "math"
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str, domain: str = "Reals") -> Tuple[bool, str]:
        """
        Eliminate quantifiers from the given formula using Mathematica.
        
        Args:
            formula: The formula with quantifiers to eliminate
            domain: The domain to use (Reals, Integers, etc.)
            
        Returns:
            Tuple of (success, result_formula)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False) as f:
            input_file = f.name
            self._write_mathematica_input(f, formula, domain)

        try:
            result = self._run_mathematica(input_file)
            os.unlink(input_file)

            if result is None:
                return False, "Timeout or error occurred"

            processed_result = self._process_output(result)
            return True, processed_result

        except Exception as e:
            logger.error(f"Error during quantifier elimination: {e}")
            try:
                os.unlink(input_file)
            except:
                pass
            return False, str(e)

    def _write_mathematica_input(self, file, formula: str, domain: str):
        """Write Mathematica input file with the appropriate commands."""
        file.write(f"formula = {formula};\n")
        file.write(f"result = Quiet[Resolve[formula, {domain}]];\n")
        file.write("Print[\"RESULT_START\"];\n")
        file.write("Print[result];\n")
        file.write("Print[\"RESULT_END\"];\n")
        file.write("Exit[];\n")
        file.flush()

    def _run_mathematica(self, input_file: str) -> Optional[str]:
        """Run Mathematica with the given input file."""
        is_timeout = [False]

        try:
            cmd = [self.math_path, "-noprompt", "-run", f"<<{input_file}"]
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            timer = None
            if self.timeout > 0:
                import threading
                timer = threading.Timer(self.timeout, self._terminate, [process, is_timeout])
                timer.start()

            stdout, stderr = process.communicate()

            if timer:
                timer.cancel()

            end_time = time.time()
            logger.debug(f"Mathematica execution time: {end_time - start_time:.2f} seconds")

            if is_timeout[0]:
                logger.warning("Mathematica execution timed out")
                return None

            if process.returncode != 0:
                logger.error(f"Mathematica execution failed with code {process.returncode}")
                logger.error(f"Stderr: {stderr}")
                return None

            return stdout

        except Exception as e:
            logger.error(f"Error executing Mathematica: {e}")
            return None

    def _terminate(self, process, is_timeout: List):
        """
        Terminates a process and sets the timeout flag to True.
        
        Args:
            process: The process to be terminated.
            is_timeout: A list containing a single boolean item. If the process exceeds 
                        the timeout limit, the boolean item will be set to True.
        """
        if process.poll() is None:
            try:
                process.terminate()
                is_timeout[0] = True
            except Exception as ex:
                logger.error(f"Error terminating Mathematica process: {ex}")
                try:
                    process.kill()
                except Exception:
                    pass

    def _process_output(self, output: str) -> str:
        """Process the output from Mathematica to extract the result formula."""
        lines = output.strip().split('\n')
        result_lines = []
        capture = False

        for line in lines:
            if "RESULT_START" in line:
                capture = True
                continue
            elif "RESULT_END" in line:
                capture = False
                continue
            elif capture:
                result_lines.append(line.strip())

        # Join the result lines and clean up
        result = ' '.join(result_lines)

        # Convert Mathematica syntax to more standard form if needed
        # This is a simple example, might need more sophisticated conversion
        result = result.replace("&&", "∧").replace("||", "∨")

        return result


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
