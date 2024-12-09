# coding: utf-8
from typing import List, Tuple

import z3
from pysmt.fnode import FNode
from pysmt.oracles import get_logic
from pysmt.shortcuts import EqualsOrIff
from pysmt.shortcuts import Not, Solver
from pysmt.shortcuts import Symbol, And
from pysmt.typing import INT, REAL, BVType, BOOL


# NOTE: both pysmt and z3 have a class "Solver"
# logger = logging.getLogger(__name__)

class Z3ToPySMTConverter:
    """Handles conversion between Z3 and PySMT expressions."""

    @staticmethod
    def to_pysmt_vars(z3vars: List[z3.ExprRef]) -> List[Symbol]:
        """
        Convert Z3 variables to PySMT variables.

        Args:
            z3vars: List of Z3 expression references

        Returns:
            List of PySMT Symbol objects

        Raises:
            NotImplementedError: If unsupported Z3 type is encountered
        """
        type_mapping = {
            z3.is_int: INT,
            z3.is_real: REAL,
            z3.is_bool: BOOL
        }

        result = []
        for var in z3vars:
            var_name = var.decl().name()

            # Handle BV type separately due to size parameter
            if z3.is_bv(var):
                result.append(Symbol(var_name, BVType(var.sort().size())))
                continue

            # Handle other types
            for type_check, pysmt_type in type_mapping.items():
                if type_check(var):
                    result.append(Symbol(var_name, pysmt_type))
                    break
            else:
                raise NotImplementedError(f"Unsupported Z3 type for variable: {var}")

        return result

    @staticmethod
    def convert(z3_formula: z3.ExprRef) -> Tuple[List[Symbol], FNode]:
        """
        Convert Z3 formula to PySMT format.

        Args:
            z3_formula: Z3 expression to convert

        Returns:
            Tuple of (PySMT variables, PySMT formula)
        """
        from arlib.utils.z3_expr_utils import get_variables

        z3_vars = get_variables(z3_formula)
        pysmt_vars = Z3ToPySMTConverter.to_pysmt_vars(z3_vars)

        # Convert formula using Z3 solver
        z3_solver = Solver(name='z3')
        pysmt_formula = z3_solver.converter.back(z3_formula)

        return pysmt_vars, pysmt_formula


def all_smt_with_pysmt(fml, keys, bound: int) -> List[List[FNode]]:
    """
    Sample multiple models satisfying the given formula.
    FIXME: use keys (the projected variables?)

    Args:
        formula: Z3 formula to solve
        keys: List of variables to include in models
        bound: Maximum number of models to generate

    Returns:
        List of models, where each model is a list of variable assignments

    Raises:
        Exception: If solver encounters an error
    """
    z3_formula = z3.And(fml)
    pysmt_vars, pysmt_formula = Z3ToPySMTConverter.convert(z3_formula)
    target_logic = get_logic(pysmt_formula)

    models = []
    try:
        with Solver(logic=target_logic) as solver:
            solver.add_assertion(pysmt_formula)

            for _ in range(bound):
                if not solver.solve():
                    break

                model = [EqualsOrIff(k, solver.get_value(k)) for k in pysmt_vars]
                models.append(model)

                # Add constraint to find different model in next iteration
                solver.add_assertion(Not(And(model)))

        return models

    except Exception as e:
        raise Exception(f"Error during model sampling: {str(e)}")
