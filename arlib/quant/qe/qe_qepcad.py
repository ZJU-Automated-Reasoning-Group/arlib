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

logger = logging.getLogger(__name__)


class QepcadQE:
    """Interface to QEPCAD for quantifier elimination."""

    def __init__(self, qepcad_path: Optional[str] = None, timeout: int = 300):
        """
        Initialize the QEPCAD QE interface.
        
        Args:
            qepcad_path: Path to the QEPCAD executable. If None, assumes 'qepcad' is in PATH.
            timeout: Timeout in seconds for QEPCAD execution.
        """
        self.qepcad_path = qepcad_path if qepcad_path else "qepcad"
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str) -> Tuple[bool, str]:
        """
        Eliminate quantifiers from the given formula using QEPCAD.
        
        Args:
            formula: The formula with quantifiers to eliminate
            
        Returns:
            Tuple of (success, result_formula)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qe', delete=False) as f:
            input_file = f.name
            self._write_qepcad_input(f, formula)

        try:
            result = self._run_qepcad(input_file)
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

    def _write_qepcad_input(self, file, formula: str):
        """Write QEPCAD input file with the appropriate commands."""
        # Convert formula to QEPCAD syntax if needed
        qepcad_formula = self._convert_to_qepcad_syntax(formula)

        file.write("[]\n")  # Start of input
        file.write(qepcad_formula + ".\n")
        file.write("finish\n")
        file.flush()

    def _run_qepcad(self, input_file: str) -> Optional[str]:
        """Run QEPCAD with the given input file."""
        is_timeout = [False]

        try:
            cmd = [self.qepcad_path, "+N500000000", "+L100000", "+H1000000"]
            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            timer = None
            if self.timeout > 0:
                import threading
                timer = threading.Timer(self.timeout, self._terminate, [process, is_timeout])
                timer.start()

            with open(input_file, 'r') as f:
                input_content = f.read()

            stdout, stderr = process.communicate(input=input_content)

            if timer:
                timer.cancel()

            end_time = time.time()
            logger.debug(f"QEPCAD execution time: {end_time - start_time:.2f} seconds")

            if is_timeout[0]:
                logger.warning("QEPCAD execution timed out")
                return None

            if process.returncode != 0:
                logger.error(f"QEPCAD execution failed with code {process.returncode}")
                logger.error(f"Stderr: {stderr}")
                return None

            return stdout

        except Exception as e:
            logger.error(f"Error executing QEPCAD: {e}")
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
                logger.error(f"Error terminating QEPCAD process: {ex}")
                try:
                    process.kill()
                except Exception:
                    pass

    def _process_output(self, output: str) -> str:
        """Process the output from QEPCAD to extract the result formula."""
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

        result = ' '.join(result_lines)
        return self._convert_from_qepcad_syntax(result)

    def _convert_to_qepcad_syntax(self, formula: str) -> str:
        """Convert a formula to QEPCAD syntax."""
        # This is a basic conversion and should be expanded based on needs
        formula = formula.replace("∧", "/\\")
        formula = formula.replace("∨", "\\/")
        formula = formula.replace("¬", "~")
        formula = formula.replace("→", "==>")
        formula = formula.replace("∀", "A")
        formula = formula.replace("∃", "E")
        return formula

    def _convert_from_qepcad_syntax(self, formula: str) -> str:
        """Convert QEPCAD output to standard mathematical notation."""
        # This is a basic conversion and should be expanded based on needs
        formula = formula.replace("/\\", "∧")
        formula = formula.replace("\\/", "∨")
        formula = formula.replace("~", "¬")
        formula = formula.replace("==>", "→")
        return formula


def eliminate_quantifiers(formula: str, qepcad_path: Optional[str] = None,
                          timeout: int = 300) -> Tuple[bool, str]:
    """
    Convenience function to eliminate quantifiers from a formula using QEPCAD.
    
    Args:
        formula: The formula with quantifiers to eliminate
        qepcad_path: Path to the QEPCAD executable
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, result_formula)
    """
    qe = QepcadQE(qepcad_path, timeout)
    return qe.eliminate_quantifiers(formula)


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
