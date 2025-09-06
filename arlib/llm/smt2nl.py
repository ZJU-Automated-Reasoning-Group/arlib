#!/usr/bin/env python3
"""Convert SMT-LIB assertions to natural language descriptions.

This script parses SMT-LIB S-expressions and converts them to human-readable
natural language descriptions.

Examples:
    python smt2nl.py "(assert (> x 5))"
    # Output: "(x > 5)"

    python smt2nl.py "(assert (and (> x 5) (<= y 10)))"
    # Output: "((x > 5) ∧ (y ≤ 10))"

    python smt2nl.py "(assert (= (+ x y) (* z 2)))"
    # Output: "((x + y) equals (z × 2))"

    python smt2nl.py "(assert (> x 5)) (assert (< y 10))"
    # Output: "(x > 5) ∧ (y < 10)"

    # Bit-vectors, strings, arrays, floats supported
    # Interactive mode: python smt2nl.py
"""

import sys
import re
from typing import Any, List, Union


class SMT2NLConverter:
    """Converts SMT-LIB S-expressions to natural language."""

    def __init__(self):
        self.ops = {
            # Core
            '=': 'equals', '!=': 'not equals', 'distinct': 'are distinct',
            'ite': 'if-then-else',

            # Arithmetic
            '>': '>', '>=': '≥', '<': '<', '<=': '≤',
            '+': '+', '-': '-', '*': '×', '/': '÷', 'mod': 'mod', 'div': 'div',
            'abs': 'abs', 'to_real': 'real', 'to_int': 'int', 'is_int': 'is_int',

            # Boolean
            'and': '∧', 'or': '∨', 'not': '¬', '=>': '⟹', 'iff': '⟺', 'xor': '⊕',

            # Arrays
            'select': '[]', 'store': 'store',

            # Bit-vectors
            'bvnot': '~', 'bvand': '&', 'bvor': '|', 'bvxor': '^',
            'bvadd': '+', 'bvsub': '-', 'bvmul': '×', 'bvudiv': '÷', 'bvurem': '%',
            'bvshl': '<<', 'bvlshr': '>>', 'bvashr': '>>>',
            'bvult': '<', 'bvule': '≤', 'bvugt': '>', 'bvuge': '≥',
            'bvslt': '<ₛ', 'bvsle': '≤ₛ', 'bvsgt': '>ₛ', 'bvsge': '≥ₛ',
            'concat': '∘', 'extract': 'extract',

            # Strings
            'str.len': 'length', 'str.++': 'concat', 'str.at': 'charAt',
            'str.substr': 'substring', 'str.contains': 'contains', 'str.prefixof': 'prefixOf',
            'str.suffixof': 'suffixOf', 'str.indexof': 'indexOf', 'str.replace': 'replace',
            'str.to_re': 'toRegex', 'str.in_re': 'matches', 're.++': 'reConcat',
            're.*': 'star', 're.+': 'plus', 're.opt': 'optional', 're.union': 'union',
            're.inter': 'intersect', 're.comp': 'complement',

            # Sets
            'union': '∪', 'intersection': '∩', 'setminus': '\\', 'subset': '⊆',
            'member': '∈', 'singleton': '{·}', 'insert': 'insert',
            # Real/Float
            'fp.abs': 'abs', 'fp.neg': '-', 'fp.add': '+', 'fp.sub': '-',
            'fp.mul': '×', 'fp.div': '÷', 'fp.sqrt': '√', 'fp.rem': 'rem',
            'fp.roundToIntegral': 'round', 'fp.min': 'min', 'fp.max': 'max',
            'fp.leq': '≤', 'fp.lt': '<', 'fp.geq': '≥', 'fp.gt': '>',
            'fp.eq': '=', 'fp.isNormal': 'isNormal', 'fp.isZero': 'isZero',
            'fp.isInfinite': 'isInf', 'fp.isNaN': 'isNaN',
            # Sequences
            'seq.len': 'length', 'seq.++': 'concat', 'seq.at': 'at',
            'seq.nth': 'nth', 'seq.extract': 'extract', 'seq.contains': 'contains',
            'seq.prefixof': 'prefixOf', 'seq.suffixof': 'suffixOf',
        }

    def parse_sexpr(self, s: str) -> Union[str, List[Any]]:
        """Parse S-expression into nested list structure."""
        s = s.strip()
        if not s.startswith('('):
            return s
        tokens = self._tokenize(s)
        return self._parse_tokens(tokens)[0]

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r'\(|\)|[^()\s]+', s)

    def _parse_tokens(self, tokens: List[str]) -> tuple:
        """Parse tokens into nested structure."""
        if not tokens:
            return None, 0

        if tokens[0] == '(':
            result, i = [], 1
            while i < len(tokens) and tokens[i] != ')':
                expr, consumed = self._parse_tokens(tokens[i:])
                result.append(expr)
                i += consumed
            return result, i + 1
        else:
            return tokens[0], 1

    def convert_expr(self, expr: Union[str, List[Any]]) -> str:
        if isinstance(expr, str):
            if expr.startswith('#b'): return f"0b{expr[2:]}"
            if expr.startswith('#x'): return f"0x{expr[2:]}"
            if expr.startswith('_'): return expr[1:]  # Remove underscore prefix
            return expr

        if not expr: return ""
        op, *args = expr

        # Special forms
        if op == 'assert': return self.convert_expr(args[0]) if args else ""
        if op == 'let':
            if len(args) >= 2:
                binds = ", ".join(f"{b[0]}={self.convert_expr(b[1])}"
                                for b in args[0] if len(b) == 2)
                return f"let {binds} in {self.convert_expr(args[1])}"
        if op == 'ite' and len(args) == 3:
            return f"({self.convert_expr(args[0])} ? {self.convert_expr(args[1])} : {self.convert_expr(args[2])})"
        if op in ['forall', 'exists'] and len(args) >= 2:
            vars_text = ", ".join(f"{v[0]}:{v[1]}" for v in args[0] if len(v) >= 2)
            return f"{'∀' if op == 'forall' else '∃'}{vars_text}. {self.convert_expr(args[1])}"

        # Extract bit-vector sizes
        if op.startswith('(_ ') and op.endswith(')'):
            parts = op[3:-1].split()
            if len(parts) >= 2:
                op = parts[0] + f"[{':'.join(parts[1:])}]"

        # Operators
        if op in self.ops:
            sym = self.ops[op]
            if len(args) == 1: return f"{sym}{self.convert_expr(args[0])}"
            if len(args) == 2:
                l, r = self.convert_expr(args[0]), self.convert_expr(args[1])
                if op in ['select']: return f"{l}[{r}]"
                if op in ['store']: return f"{l}[{r} := {self.convert_expr(args[2])}]" if len(args) > 2 else f"{l}[{r}]"
                return f"({l} {sym} {r})"
            # N-ary
            arg_strs = [self.convert_expr(a) for a in args]
            if op in ['and', 'or', '+', '*', 'bvadd', 'bvmul']:
                return f"({f' {sym} '.join(arg_strs)})"
            return f"{sym}({', '.join(arg_strs)})"

        # Function calls
        arg_strs = [self.convert_expr(a) for a in args]
        return f"{op}({', '.join(arg_strs)})" if args else op

    def convert(self, smt_text: str) -> str:
        try:
            smt_text = re.sub(r';.*$', '', smt_text, flags=re.MULTILINE).strip()
            return (self._convert_multiple(smt_text) if self._has_multiple_expressions(smt_text)
                   else self.convert_expr(self.parse_sexpr(smt_text)))
        except Exception as e:
            return f"Error: {e}"

    def _has_multiple_expressions(self, smt_text: str) -> bool:
        tokens, depth, count = self._tokenize(smt_text), 0, 0
        for t in tokens:
            if t == '(' and depth == 0: count += 1
            depth += 1 if t == '(' else -1 if t == ')' else 0
        return count > 1

    def _convert_multiple(self, smt_text: str) -> str:
        tokens, exprs, i = self._tokenize(smt_text), [], 0
        while i < len(tokens):
            if tokens[i] == '(':
                depth, start = 1, i
                i += 1
                while i < len(tokens) and depth > 0:
                    depth += 1 if tokens[i] == '(' else -1 if tokens[i] == ')' else 0
                    i += 1
                expr_text = ' '.join(tokens[start:i])
                exprs.append(self.convert_expr(self.parse_sexpr(expr_text)))
            else:
                exprs.append(self.convert_expr(tokens[i]))
                i += 1
        return ' ∧ '.join(exprs) if len(exprs) > 1 else exprs[0] if exprs else ""


def main():
    """Main function for command-line usage."""
    converter = SMT2NLConverter()

    if len(sys.argv) > 1:
        result = converter.convert(sys.argv[1])
        print(result)
    else:
        print("SMT-LIB to Natural Language Converter")
        print("Enter SMT-LIB expressions (Ctrl+C to exit):")
        print()

        try:
            while True:
                smt_text = input("SMT> ").strip()
                if smt_text:
                    result = converter.convert(smt_text)
                    print(f"NL>  {result}")
                    print()
        except KeyboardInterrupt:
            print("\nGoodbye!")


if __name__ == "__main__":
    main()
