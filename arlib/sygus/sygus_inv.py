"""
Solving SyGuS (Inverse Synthesis) problems
"""

import re
from typing import Dict, Optional, Any
import z3
from arlib.utils.z3_plus_smtlib_solver import Z3SolverPlus


class SygusInv:
    """
    Class for solving SyGuS invariant synthesis problems.
    
    Invariant synthesis is a common verification task where we need to find
    an invariant formula that satisfies certain conditions - typically used 
    in program verification to find loop invariants.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the SyGuS invariant synthesis solver.
        
        Args:
            debug: Whether to enable debug output
        """
        self.debug = debug
        self.solver = Z3SolverPlus(debug=debug)
    
    def parse_sygus_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse a SyGuS file in .sl format and extract relevant information.
        
        Args:
            filepath: Path to the SyGuS file
            
        Returns:
            Dictionary containing parsed components of the SyGuS problem
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract logic
        logic = None
        logic_match = re.search(r'\(set-logic\s+([A-Z_]+)\)', content)
        if logic_match:
            logic = logic_match.group(1)
        
        # Extract function name and variables
        synth_inv_match = re.search(r'\(synth-inv\s+([a-zA-Z0-9_]+)\s+\(\((.*?)\)\)', content, re.DOTALL)
        if not synth_inv_match:
            raise ValueError("Could not find synth-inv declaration in the file")
        
        inv_fun_name = synth_inv_match.group(1)
        vars_decl = synth_inv_match.group(2)
        
        # Parse variable declarations
        var_list = []
        var_sorts = {}
        
        # Split the variable declarations and process each
        for var_decl in re.findall(r'([a-zA-Z0-9_]+)\s+([^)]+)', vars_decl):
            var_name, sort_str = var_decl
            var_list.append(var_name)
            var_sorts[var_name] = sort_str.strip()
        
        # Extract pre, trans, and post functions
        pre_match = re.search(r'\(define-fun\s+pre_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)\s*\(define-fun', content, re.DOTALL)
        if pre_match:
            pre_body = pre_match.group(2).strip()
        else:
            # Try different pattern if the first one didn't match
            pre_match = re.search(r'\(define-fun\s+pre_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)(?:\s*\n|\s*$)', content, re.DOTALL)
            if not pre_match:
                raise ValueError("Could not find pre_fun in the file")
            pre_body = pre_match.group(2).strip()
            
        # Extract trans function
        trans_match = re.search(r'\(define-fun\s+trans_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)\s*\(define-fun', content, re.DOTALL)
        if trans_match:
            trans_body = trans_match.group(2).strip()
        else:
            # Try different pattern
            trans_match = re.search(r'\(define-fun\s+trans_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)(?:\s*\n|\s*$)', content, re.DOTALL)
            if not trans_match:
                raise ValueError("Could not find trans_fun in the file")
            trans_body = trans_match.group(2).strip()
            
        # Extract post function
        post_match = re.search(r'\(define-fun\s+post_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)(?:\s*\n|\s*$)', content, re.DOTALL)
        if not post_match:
            post_match = re.search(r'\(define-fun\s+post_fun\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)\s*\(', content, re.DOTALL)
            if not post_match:
                raise ValueError("Could not find post_fun in the file")
        post_body = post_match.group(2).strip()
        
        return {
            'logic': logic,
            'inv_fun_name': inv_fun_name,
            'var_list': var_list,
            'var_sorts': var_sorts,
            'pre_body': pre_body,
            'trans_body': trans_body,
            'post_body': post_body
        }
    
    def synthesize_invariant(self, filepath: str) -> Optional[str]:
        """
        Synthesize an invariant function for the SyGuS problem in the given file.
        
        Args:
            filepath: Path to the SyGuS invariant synthesis file
            
        Returns:
            The synthesized invariant as an SMT-LIB2 string, or None if synthesis failed
        """
        # Parse the SyGuS file
        problem = self.parse_sygus_file(filepath)
        
        # We'll create a full SyGuS string to pass to the solver
        sygus_content = f"(set-logic {problem['logic']})\n\n"
        sygus_content += f"(synth-inv {problem['inv_fun_name']} ("
        for var in problem['var_list']:
            sygus_content += f"({var} {problem['var_sorts'][var]}) "
        sygus_content += "))\n\n"
        
        # Add the pre, trans, and post functions
        pre_args = ' '.join(f"({var} {problem['var_sorts'][var]})" for var in problem['var_list'])
        sygus_content += f"(define-fun pre_fun ({pre_args}) Bool\n    {problem['pre_body']})\n"
        
        # For trans_fun, we need both current and next-state variables
        trans_args = ' '.join(f"({var} {problem['var_sorts'][var]})" for var in problem['var_list'])
        trans_args += ' ' + ' '.join(f"({var}! {problem['var_sorts'][var]})" for var in problem['var_list'])
        sygus_content += f"(define-fun trans_fun ({trans_args}) Bool\n    {problem['trans_body']})\n"
        
        sygus_content += f"(define-fun post_fun ({pre_args}) Bool\n    {problem['post_body']})\n\n"
        
        # Add the inv-constraint
        sygus_content += f"(inv-constraint {problem['inv_fun_name']} pre_fun trans_fun post_fun)\n\n"
        sygus_content += "(check-synth)\n"
        
        # Use CVC5 to solve the invariant synthesis problem
        if self.debug:
            print("Generated SyGuS problem:")
            print(sygus_content)
        
        result = self._invoke_cvc5_sygus_inv(sygus_content, problem['logic'])
        
        if self.debug:
            print("SyGuS result:", result)
            
        return result
    
    def _invoke_cvc5_sygus_inv(self, sygus_content: str, logic: str) -> Optional[str]:
        """
        Invoke CVC5 to solve the SyGuS invariant synthesis problem.
        
        Args:
            sygus_content: The SyGuS problem in string format
            logic: The logic to use
            
        Returns:
            The synthesized invariant as an SMT-LIB2 string, or None if synthesis failed
        """
        import tempfile
        import subprocess
        import os
        from arlib.global_params.paths import global_config
        
        # Get CVC5 path
        cvc5_path = global_config.get_solver_path("cvc5")
        if not cvc5_path:
            if self.debug:
                print("CVC5 not found in global config. Using 'cvc5' command from PATH.")
            cvc5_path = "cvc5"
        
        # Create a temporary file with the SyGuS content
        with tempfile.NamedTemporaryFile(suffix='.sl', mode='w', delete=False) as tmp:
            tmp.write(sygus_content)
            tmp_path = tmp.name
        
        try:
            # Run CVC5 with appropriate flags for invariant synthesis
            # Note: Removed --sygus-inv-syn flag which is not available in the CVC5 version
            cmd = [cvc5_path, "--lang=sygus2", "--produce-models", tmp_path]
            
            if self.debug:
                print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Set a timeout of 60 seconds
            )
            
            if result.returncode != 0:
                if self.debug:
                    print(f"CVC5 error (return code {result.returncode}):")
                    print(result.stderr)
                return None
            
            # Parse the output to extract the synthesized invariant
            output = result.stdout.strip()
            
            if self.debug:
                print("CVC5 output:")
                print(output)
            
            # Extract the synthesized invariant from CVC5's output
            if "unsat" in output.lower():
                return None
            
            inv_match = re.search(r'\(define-fun\s+([a-zA-Z0-9_]+)\s+\(\((.*?)\)\)\s+Bool\s+(.*?)\)', 
                                output, re.DOTALL)
            
            if inv_match:
                inv_fun_name = inv_match.group(1)
                inv_args = inv_match.group(2)
                inv_body = inv_match.group(3).strip()
                
                # Format the invariant as a clean SMT-LIB2 string
                return f"(define-fun {inv_fun_name} (({inv_args})) Bool {inv_body})"
            
            return output
            
        except subprocess.TimeoutExpired:
            if self.debug:
                print("CVC5 timeout expired")
            return None
        except Exception as e:
            if self.debug:
                print(f"Error invoking CVC5: {e}")
            return None
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

