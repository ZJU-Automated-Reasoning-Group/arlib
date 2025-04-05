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

logger = logging.getLogger(__name__)


class RedlogQE:
    """Interface to Redlog for quantifier elimination."""

    def __init__(self, redlog_path: Optional[str] = None, timeout: int = 300):
        """
        Initialize the Redlog QE interface.
        
        Args:
            redlog_path: Path to the Redlog executable. If None, assumes 'redlog' is in PATH.
            timeout: Timeout in seconds for Redlog execution.
        """
        self.redlog_path = redlog_path if redlog_path else "redlog"
        self.timeout = timeout

    def eliminate_quantifiers(self, formula: str, logic: str = "real") -> Tuple[bool, str]:
        """
        Eliminate quantifiers from the given formula using Redlog.
        
        Args:
            formula: The formula with quantifiers to eliminate
            logic: The logic to use (real, integer, etc.)
            
        Returns:
            Tuple of (success, result_formula)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.red', delete=False) as f:
            input_file = f.name
            self._write_redlog_input(f, formula, logic)

        try:
            result = self._run_redlog(input_file)
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

    def _write_redlog_input(self, file, formula: str, logic: str):
        """Write Redlog input file with the appropriate commands."""
        if logic.lower() == "real":
            file.write("load_package redlog;\n")
            file.write("rlset ofsf;\n")  # Ordered fields (real closed fields)
        elif logic.lower() == "integer":
            file.write("load_package redlog;\n")
            file.write("rlset pasf;\n")  # Presburger arithmetic
        else:
            file.write("load_package redlog;\n")
            file.write(f"rlset {logic};\n")

        file.write(f"formula := {formula};\n")
        file.write("result := rlqe formula;\n")
        file.write("print_with result;\n")
        file.write("quit;\n")
        file.flush()

    def _run_redlog(self, input_file: str) -> Optional[str]:
        """Run Redlog with the given input file."""
        is_timeout = [False]

        try:
            cmd = [self.redlog_path, "-w", input_file]
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
            logger.debug(f"Redlog execution time: {end_time - start_time:.2f} seconds")

            if is_timeout[0]:
                logger.warning("Redlog execution timed out")
                return None

            if process.returncode != 0:
                logger.error(f"Redlog execution failed with code {process.returncode}")
                logger.error(f"Stderr: {stderr}")
                return None

            return stdout

        except Exception as e:
            logger.error(f"Error executing Redlog: {e}")
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
                logger.error(f"Error terminating Redlog process: {ex}")
                try:
                    process.kill()
                except Exception:
                    pass

    def _process_output(self, output: str) -> str:
        """Process the output from Redlog to extract the result formula."""
        lines = output.strip().split('\n')
        result_lines = []
        capture = False

        for line in lines:
            if "result :=" in line:
                capture = True
                # Extract the part after ":="
                parts = line.split(":=", 1)
                if len(parts) > 1:
                    result_lines.append(parts[1].strip())
            elif capture and line and not line.startswith("quit"):
                result_lines.append(line.strip())

        # Join the result lines and clean up
        result = ' '.join(result_lines)
        # Remove trailing semicolon if present
        if result.endswith(';'):
            result = result[:-1].strip()

        return result


def eliminate_quantifiers(formula: str, logic: str = "real",
                          redlog_path: Optional[str] = None,
                          timeout: int = 300) -> Tuple[bool, str]:
    """
    Convenience function to eliminate quantifiers from a formula using Redlog.
    
    Args:
        formula: The formula with quantifiers to eliminate
        logic: The logic to use (real, integer, etc.)
        redlog_path: Path to the Redlog executable
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, result_formula)
    """
    qe = RedlogQE(redlog_path, timeout)
    return qe.eliminate_quantifiers(formula, logic)


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
