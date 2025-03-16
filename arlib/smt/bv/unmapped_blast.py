"""
Converting QF_BV to SAT without tracking the mapppings between BV variables and SAT variables.

This module provides functionality to convert QF_BV (Quantifier-Free Bit-Vector) formulas 
to SAT problems using Z3's tactics and PySAT solvers.
"""

import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from typing import Union, Optional
from arlib.utils.types import SolverResult

# Keep existing preamble but add type annotation
qfbv_preamble: z3.Tactic = z3.AndThen(
    z3.With('simplify', flat_and_or=False),
    z3.With('propagate-values', flat_and_or=False),
    z3.Tactic('elim-uncnstr'),
    z3.With('solve-eqs', solve_eqs_max_occs=2),
    z3.Tactic('reduce-bv-size'),
    z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
            local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
    z3.Tactic('max-bv-sharing'),
    z3.Tactic('ackermannize_bv'),
    z3.Tactic('bit-blast'),
    z3.Tactic('tseitin-cnf')
)

qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)


def qfbv_to_sat(fml: z3.ExprRef, solver_name: str = "minisat22") -> SolverResult:
    """
    Convert a QF_BV formula to SAT and solve it.

    Args:
        fml: Z3 bit-vector formula to solve
        solver_name: Name of the PySAT solver to use (default: minisat22)

    Returns:
        SolverResult indicating SAT/UNSAT

    Raises:
        z3.Z3Exception: If formula conversion fails
        ValueError: If formula is not a bit-vector formula
    """
    # Type checking
    if not isinstance(fml, z3.ExprRef):
        raise ValueError("Input must be a Z3 expression")
    
    # Check if formula contains only bit-vector operations
    if not all(z3.is_bv(arg) for arg in z3.get_vars(fml)):
        raise ValueError("Formula must contain only bit-vector operations")

    try:
        # Apply tactics and convert to CNF
        after_simp = qfbv_tactic(fml).as_expr()
        
        # Handle trivial cases
        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        elif z3.is_true(after_simp):
            return SolverResult.SAT

        # Convert to CNF
        g = z3.Goal()
        g.add(after_simp)
        pos = CNF(from_string=g.dimacs())

        # Solve using PySAT
        with Solver(name=solver_name, bootstrap_with=pos) as aux:
            if aux.solve():
                return SolverResult.SAT
            return SolverResult.UNSAT

    except z3.Z3Exception as e:
        raise z3.Z3Exception(f"Z3 conversion failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"SAT solving failed: {str(e)}")


def demo(verbose: bool = False) -> None:
    """
    Demonstrate the QF_BV to SAT conversion with a simple example.
    
    Args:
        verbose: Enable verbose output from Z3 (default: False)
    """
    if verbose:
        z3.set_param("verbose", 15)
    
    # Example 1: Simple arithmetic
    x, y = z3.BitVecs("x y", 6)
    fml1 = z3.And(x + y == 8, x - y == 2)
    print("Example 1:", qfbv_to_sat(fml1))

    # Example 2: More complex operations
    a, b = z3.BitVecs("a b", 4)
    fml2 = z3.And(z3.ULT(a, b), a & b == 0)
    print("Example 2:", qfbv_to_sat(fml2))


if __name__ == "__main__":
    demo()
