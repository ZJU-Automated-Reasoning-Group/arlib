"""
Utilities for parsing and manipulating S-expressions.

S-expressions are symbolic expressions that represent nested lists of atoms,
commonly used in Lisp-like languages. This module provides tools to parse
string representations of S-expressions into Python data structures.

Example:
    "(+ 1 (* 2 3))" -> ['+', 1, ['*', 2, 3]]
"""

from typing import Union, List, Optional
from dataclasses import dataclass

# Type definitions
Symbol = str
Number = Union[int, float]
Atom = Union[Symbol, Number]
SExpr = Union[Atom, List['SExpr']]

@dataclass
class ParseError(Exception):
    """Exception raised for S-expression parsing errors."""
    message: str
    position: int
    expression: str

    def __str__(self) -> str:
        return f"{self.message} at position {self.position}: {self.expression}"

def tokenize(expression: str) -> list[str]:
    """
    Convert a string into a list of S-expression tokens.

    Args:
        expression: String containing S-expression

    Returns:
        List of tokens (parentheses, atoms, etc.)

    Example:
        >>> tokenize("(+ 1 2)")
        ['(', '+', '1', '2', ')']
    """
    return (expression.replace('(', ' ( ')
                     .replace(')', ' ) ')
                     .replace('" "', 'space')
                     .split())

def parse(expression: str) -> Optional[SExpr]:
    """
    Parse a string into an S-expression.

    Args:
        expression: String containing S-expression

    Returns:
        Parsed S-expression as nested Python lists/atoms

    Raises:
        ParseError: If the expression is malformed

    Example:
        >>> parse("(+ 1 (* 2 3))")
        ['+', 1, ['*', 2, 3]]
    """
    tokens = tokenize(expression)
    if not tokens:
        return None
    try:
        result, remaining = _parse_tokens(tokens, 0)
        if remaining:
            raise ParseError("Unexpected trailing tokens", len(tokens) - len(remaining), expression)
        return result
    except (IndexError, ValueError) as e:
        raise ParseError(str(e), len(tokens), expression)

def _parse_tokens(tokens: list[str], depth: int) -> tuple[SExpr, list[str]]:
    """
    Recursively parse a list of tokens into an S-expression.

    Args:
        tokens: List of remaining tokens to parse
        depth: Current nesting depth

    Returns:
        Tuple of (parsed expression, remaining tokens)
    """
    if not tokens:
        raise ValueError("Unexpected end of expression")

    token = tokens[0]
    remaining = tokens[1:]

    if token == '(':
        lst: List[SExpr] = []
        while remaining and remaining[0] != ')':
            expr, remaining = _parse_tokens(remaining, depth + 1)
            lst.append(expr)
        if not remaining:
            raise ValueError("Missing closing parenthesis")
        return lst, remaining[1:]  # Skip closing paren

    elif token == ')':
        raise ValueError("Unexpected closing parenthesis")

    else:
        return parse_atom(token), remaining

def parse_atom(token: str) -> Atom:
    """
    Convert a token string into an atomic value (number or symbol).

    Args:
        token: String token to parse

    Returns:
        Parsed atomic value
    """
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


def parse_sexpr_string(program: str) -> Optional[SExpr]:
    """
    Read an S-expression from a string.
    This is a legacy API wrapper around parse().

    Args:
        program: String containing S-expression

    Returns:
        Parsed S-expression as nested Python lists/atoms
    """
    return parse(program)
