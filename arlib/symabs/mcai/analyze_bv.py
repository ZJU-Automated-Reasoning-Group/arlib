"""
FIXME: generated by LLM
Bit-Vector Expression Extractor for Z3 Formulas

This module provides functionality to:
1. Extract all bit-vector expressions from Z3 formulas
2. Classify bit-vector operations
3. Analyze bit-vector expression properties
4. Track expression dependencies
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Dict

import z3


class BVExprType(Enum):
    """Classification of bit-vector expression types"""
    CONSTANT = "constant"
    VARIABLE = "variable"
    ARITHMETIC = "arithmetic"
    LOGICAL = "logical"
    COMPARISON = "comparison"
    SHIFT = "shift"
    EXTRACT = "extract"
    EXTEND = "extend"
    CONCAT = "concat"
    UNKNOWN = "unknown"


@dataclass
class BVExprInfo:
    """
    Information about a bit-vector expression

    Attributes:
        expr: The Z3 expression
        expr_type: Type of the expression
        width: Bit-width of the expression
        variables: Set of variables used in the expression
        complexity: Measure of expression complexity (operation depth)
    """
    expr: z3.ExprRef
    expr_type: BVExprType
    width: int
    variables: Set[str] = field(default_factory=set)
    complexity: int = 0

    def __hash__(self):
        return self.expr.__hash__()


class BVExprExtractor:
    """
    Extracts and analyzes bit-vector expressions from Z3 formulas

    Args:
        track_dependencies: Whether to track variable dependencies
        analyze_complexity: Whether to analyze expression complexity
    """

    def __init__(self, track_dependencies: bool = True, analyze_complexity: bool = True):
        self.track_dependencies = track_dependencies
        self.analyze_complexity = analyze_complexity
        self.expressions: Set[BVExprInfo] = set()
        self.var_dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Mapping of Z3 operations to expression types
        self._op_type_map = {
            # Arithmetic operations
            z3.Z3_OP_BADD: BVExprType.ARITHMETIC,
            z3.Z3_OP_BSUB: BVExprType.ARITHMETIC,
            z3.Z3_OP_BMUL: BVExprType.ARITHMETIC,
            z3.Z3_OP_BSDIV: BVExprType.ARITHMETIC,
            z3.Z3_OP_BUDIV: BVExprType.ARITHMETIC,
            z3.Z3_OP_BSREM: BVExprType.ARITHMETIC,
            z3.Z3_OP_BUREM: BVExprType.ARITHMETIC,
            z3.Z3_OP_BSMOD: BVExprType.ARITHMETIC,

            # Logical operations
            z3.Z3_OP_BAND: BVExprType.LOGICAL,
            z3.Z3_OP_BOR: BVExprType.LOGICAL,
            z3.Z3_OP_BNOT: BVExprType.LOGICAL,
            z3.Z3_OP_BXOR: BVExprType.LOGICAL,
            z3.Z3_OP_BNAND: BVExprType.LOGICAL,
            z3.Z3_OP_BNOR: BVExprType.LOGICAL,
            z3.Z3_OP_BXNOR: BVExprType.LOGICAL,

            # Comparison operations
            z3.Z3_OP_ULEQ: BVExprType.COMPARISON,
            z3.Z3_OP_SLEQ: BVExprType.COMPARISON,
            z3.Z3_OP_UGEQ: BVExprType.COMPARISON,
            z3.Z3_OP_SGEQ: BVExprType.COMPARISON,
            z3.Z3_OP_ULT: BVExprType.COMPARISON,
            z3.Z3_OP_SLT: BVExprType.COMPARISON,
            z3.Z3_OP_UGT: BVExprType.COMPARISON,
            z3.Z3_OP_SGT: BVExprType.COMPARISON,

            # Shift operations
            z3.Z3_OP_BASHR: BVExprType.SHIFT,
            z3.Z3_OP_BLSHR: BVExprType.SHIFT,
            z3.Z3_OP_BSHL: BVExprType.SHIFT,
            z3.Z3_OP_ROTATE_LEFT: BVExprType.SHIFT,
            z3.Z3_OP_ROTATE_RIGHT: BVExprType.SHIFT,

            # Extract/Extend operations
            z3.Z3_OP_EXTRACT: BVExprType.EXTRACT,
            z3.Z3_OP_SIGN_EXT: BVExprType.EXTEND,
            z3.Z3_OP_ZERO_EXT: BVExprType.EXTEND,

            # Concatenation
            z3.Z3_OP_CONCAT: BVExprType.CONCAT,
        }

    def _get_expr_type(self, expr: z3.ExprRef) -> BVExprType:
        """
        Determine the type of a bit-vector expression

        Args:
            expr: Z3 expression to analyze

        Returns:
            BVExprType: Type of the expression
        """
        if z3.is_bv_value(expr):
            return BVExprType.CONSTANT
        elif z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            return BVExprType.VARIABLE
        elif isinstance(expr, z3.BitVecRef):
            op = expr.decl().kind()
            return self._op_type_map.get(op, BVExprType.UNKNOWN)
        return BVExprType.UNKNOWN

    def _calculate_complexity(self, expr: z3.ExprRef) -> int:
        """
        Calculate the complexity of an expression based on its depth and operations

        Args:
            expr: Z3 expression to analyze

        Returns:
            int: Complexity score
        """
        if z3.is_const(expr) or z3.is_bv_value(expr):
            return 1

        complexity = 1
        for child in expr.children():
            complexity += self._calculate_complexity(child)

        # Add additional complexity for certain operations
        op_type = self._get_expr_type(expr)
        if op_type in {BVExprType.ARITHMETIC, BVExprType.LOGICAL}:
            complexity += 1
        elif op_type in {BVExprType.EXTRACT, BVExprType.EXTEND}:
            complexity += 2

        return complexity

    def _extract_variables(self, expr: z3.ExprRef) -> Set[str]:
        """
        Extract all variables from an expression

        Args:
            expr: Z3 expression to analyze

        Returns:
            Set[str]: Set of variable names
        """
        variables = set()

        def visit(e):
            if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                variables.add(str(e))
            for child in e.children():
                visit(child)

        visit(expr)
        return variables

    def _update_dependencies(self, expr: z3.ExprRef, variables: Set[str]):
        """
        Update the dependency graph for variables

        Args:
            expr: Z3 expression to analyze
            variables: Set of variables in the expression
        """
        if len(variables) > 1:
            for var in variables:
                self.var_dependencies[var].update(variables - {var})

    def extract_expressions(self, formula: z3.ExprRef) -> Set[BVExprInfo]:
        """
        Extract and analyze all bit-vector expressions from a formula

        Args:
            formula: Z3 formula to analyze

        Returns:
            Set[BVExprInfo]: Set of extracted expression information
        """
        self.expressions.clear()
        self.var_dependencies.clear()

        def visit(expr):
            if not isinstance(expr, z3.BitVecRef):
                for child in expr.children():
                    visit(child)
                return

            expr_type = self._get_expr_type(expr)
            width = expr.size()
            variables = self._extract_variables(expr)

            complexity = (
                self._calculate_complexity(expr)
                if self.analyze_complexity
                else 0
            )

            expr_info = BVExprInfo(
                expr=expr,
                expr_type=expr_type,
                width=width,
                variables=variables,
                complexity=complexity
            )

            self.expressions.add(expr_info)

            if self.track_dependencies:
                self._update_dependencies(expr, variables)

            for child in expr.children():
                visit(child)

        visit(formula)
        return self.expressions

    def get_expressions_by_type(self, expr_type: BVExprType) -> Set[BVExprInfo]:
        """
        Get all expressions of a specific type

        Args:
            expr_type: Type of expressions to retrieve

        Returns:
            Set[BVExprInfo]: Matching expressions
        """
        return {info for info in self.expressions if info.expr_type == expr_type}

    def get_variable_dependencies(self, variable: str) -> Set[str]:
        """
        Get all variables that a given variable depends on

        Args:
            variable: Variable name to check

        Returns:
            Set[str]: Set of dependent variables
        """
        return self.var_dependencies.get(variable, set())

    def get_complex_expressions(self, threshold: int = 5) -> Set[BVExprInfo]:
        """
        Get expressions above a complexity threshold

        Args:
            threshold: Minimum complexity score

        Returns:
            Set[BVExprInfo]: Complex expressions
        """
        return {info for info in self.expressions if info.complexity > threshold}


def analyze_formula(formula: z3.ExprRef) -> None:
    """
    Analyze and print information about bit-vector expressions in a formula

    Args:
        formula: Z3 formula to analyze
    """
    extractor = BVExprExtractor()
    expressions = extractor.extract_expressions(formula)

    print(f"Found {len(expressions)} bit-vector expressions:")

    # Print statistics by type
    for expr_type in BVExprType:
        type_exprs = extractor.get_expressions_by_type(expr_type)
        if type_exprs:
            print(f"\n{expr_type.value}: {len(type_exprs)}")
            for info in type_exprs:
                print(f"  - {info.expr} (width: {info.width}, complexity: {info.complexity})")

    # Print variable dependencies
    print("\nVariable Dependencies:")
    for var, deps in extractor.var_dependencies.items():
        if deps:
            print(f"  {var} depends on: {', '.join(deps)}")

    # Print complex expressions
    complex_exprs = extractor.get_complex_expressions()
    if complex_exprs:
        print("\nComplex Expressions:")
        for info in complex_exprs:
            print(f"  - {info.expr} (complexity: {info.complexity})")


# Example usage
def main():
    """Example usage of the bit-vector expression extractor"""
    # Create some example bit-vector variables and expressions
    x, y, z = z3.BitVecs('x y z', 8)

    # Create a formula with various bit-vector operations
    formula = z3.And(
        x + y == z,
        z & x == y,
        z >> 2 == x,
        z3.Extract(7, 4, x) == z3.Extract(3, 0, y),
        z3.Concat(x, y) == z3.ZeroExt(8, z)
    )

    # Analyze the formula
    analyze_formula(formula)


if __name__ == '__main__':
    main()