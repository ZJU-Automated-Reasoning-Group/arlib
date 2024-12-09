"""
Use SyGUS for Abduction

SyGUS (Syntax-Guided Synthesis) is a formalism for specifying synthesis problems.
It is used to automatically generate programs that satisfy a given specification.
In the context of abduction, SyGUS can be used to find an explanation for a given observation
by synthesizing a formula that satisfies the constraints of the problem.

1. CVC5 has a built-in engine for SyGuS-based abduction
2. Use the SyGuS engine of CVC5 to build one (..)
"""

import tempfile
from typing import List, Set

from z3 import *

from arlib.global_params import global_config


def get_sort(var: ExprRef) -> str:
    """
    Get the sort/type of a Z3 expression in SyGUS format.

    Args:
        var: Z3 expression

    Returns:
        String representation of the sort in SyGUS format
    """
    sort = var.sort()
    if sort.kind() == Z3_BOOL_SORT:
        return "Bool"
    elif sort.kind() == Z3_INT_SORT:
        return "Int"
    elif sort.kind() == Z3_REAL_SORT:
        return "Real"
    elif sort.kind() == Z3_BV_SORT:
        return f"(_ BitVec {sort.size()})"
    else:
        raise ValueError(f"Unsupported sort: {sort}")


class SyGUSGrammar:
    def __init__(self, supported_sorts: Set[str] = None):
        if supported_sorts is None:
            supported_sorts = {"Int", "Bool", "Real"}

        self.supported_sorts = supported_sorts
        self.rules = {
            'Start': ['Bool'],
            'BoolExpr': [
                'BoolConst',
                'BoolVar',
                '(and BoolExpr BoolExpr)',
                '(or BoolExpr BoolExpr)',
                '(not BoolExpr)',
                '(>= IntExpr IntExpr)',
                '(<= IntExpr IntExpr)',
                '(= IntExpr IntExpr)'
            ],
            'BoolConst': ['true', 'false'],
            'IntExpr': [
                '0',
                '1',
                'IntVar',
                '(+ IntExpr IntExpr)',
                '(- IntExpr IntExpr)',
                '(* IntExpr IntExpr)'
            ]
        }

    def to_sygus_format(self) -> str:
        """Convert grammar rules to SyGUS format"""
        result = []
        # First declare the starting non-terminal
        result.append('(Start Bool ((BoolExpr)))')
        # Then declare other non-terminals
        result.append('(BoolExpr Bool (')
        result.extend([f'  {prod}' for prod in self.rules['BoolExpr']])
        result.append('))')
        result.append('(BoolConst Bool (true false))')
        result.append('(IntExpr Int (')
        result.extend([f'  {prod}' for prod in self.rules['IntExpr']])
        result.append('))')
        return '\n'.join(result)


class SyGUSAbduction:
    def __init__(self, logic: str = 'LIA'):
        self.logic = logic
        supported_sorts = {"Int", "Bool"} if logic in ['LIA', 'NIA'] else {"Real", "Bool"}
        self.grammar = SyGUSGrammar(supported_sorts)

    def generate_sygus_file(self,
                            precondition: BoolRef,
                            postcondition: BoolRef,
                            variables: List[ExprRef]) -> str:
        """Generate SyGUS specification file for abduction problem."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sy', delete=False) as f:
            # Write header
            f.write(f'(set-logic {self.logic})\n\n')

            # Declare variables with their types
            for var in variables:
                sort = get_sort(var)
                f.write(f'(declare-var {var} {sort})\n')

            # Define synthesis function
            f.write('\n(synth-fun explanation (')
            var_decls = ' '.join([f'({var} {get_sort(var)})' for var in variables])
            f.write(f'{var_decls}) Bool\n')

            # Write grammar
            f.write('  (\n')
            f.write(self.grammar.to_sygus_format())
            f.write('\n  )\n)\n\n')

            # Write constraints
            pre = str(precondition).replace('And', 'and').replace('Or', 'or').replace('Not', 'not')
            post = str(postcondition).replace('And', 'and').replace('Or', 'or').replace('Not', 'not')
            f.write(f'(constraint (=> (and explanation {pre}) {post}))\n')
            f.write('(constraint (not (= explanation false)))\n')

            # Write check-synth command
            f.write('\n(check-synth)\n')

            return f.name


def sygus_via_cvc5(sygus_file: str, cvc5_bin: str) -> str:
    """Use CVC5 solver for SyGUS problems"""
    import subprocess

    try:
        result = subprocess.run(
            [cvc5_bin, "--lang=sygus2", "--sygus-stream", sygus_file],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        return "synthesis failed"

    except subprocess.TimeoutExpired:
        return "timeout"
    except subprocess.CalledProcessError as e:
        return f"error: {e}"
    except Exception as e:
        return f"unexpected error: {e}"


def sygus_via_cvc5(sygus_file: str, cvc5_bin: str) -> str:
    """
    Use CVC5 solver for SyGUS problems

    Args:
        sygus_file: Path to SyGUS specification file
        cvc5_bin: Path to CVC5 binary

    Returns:
        Synthesized formula as string
    """
    import subprocess

    try:
        # Call CVC5 with appropriate arguments
        result = subprocess.run(
            [cvc5_bin, "--lang=sygus2", sygus_file],
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # Timeout after 30 seconds
        )

        # Process and return result
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        return "synthesis failed"

    except subprocess.TimeoutExpired:
        return "timeout"
    except subprocess.CalledProcessError as e:
        return f"error: {e}"
    except Exception as e:
        return f"unexpected error: {e}"


def demo():
    """Demonstrate SyGUS-based abduction"""
    # Create variables
    x, y = Ints('x y')

    # Define precondition and postcondition
    precondition = And(x > 0, y > 0)
    postcondition = x + y > 5

    # Create abduction instance
    abduction = SyGUSAbduction()

    # Generate SyGUS file
    sygus_file = abduction.generate_sygus_file(precondition, postcondition, [x, y])

    print("Generated SyGUS file:")
    with open(sygus_file, 'r') as f:
        print(f.read())

    # Try synthesis with CVC5 (if available)
    cvc5_path = global_config.get_solver_path("cvc5")
    if os.path.exists(cvc5_path):
        result = sygus_via_cvc5(sygus_file, cvc5_path)
        print(f"\nSynthesis result: {result}")
    else:
        print("\nCVC5 binary not found. Please install CVC5 and update the path.")

    # Cleanup
    # os.unlink(sygus_file)


if __name__ == "__main__":
    demo()
