"""
Use SyGUS for Abduction

SyGUS (Syntax-Guided Synthesis) is a formalism for specifying synthesis problems.
It is used to automatically generate programs that satisfy a given specification.
In the context of abduction, SyGUS can be used to find an explanation for a given observation
by synthesizing a formula that satisfies the constraints of the problem.
"""

from z3 import *

# Define the SyGUS engine for LIA problems
def sygus_engine_lia(constraints, variables, grammar):
    # Create a solver for LIA problems
    raise NotADirectoryError()


def sygus_via_cvc5(sygus_file, cvc5_bin):
    # Import the necessary libraries
    import subprocess
    # Call the CVC5 binary with the SyGUS file as input
    result = subprocess.run([cvc5_bin, "--lang=sygus2", sygus_file], capture_output=True, text=True)
    # Return the result
    return result.stdout.strip()



