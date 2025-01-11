from z3 import *
import re
from typing import List, Any


class SMTLIBParser:
    def __init__(self):
        self.solver = Solver()
        self.variables = {}
        self.logic = None
        # Stack of constraints for each scope level
        # First list (index 0) contains global constraints
        self.constraints_stack: List[List[Any]] = [[]]

    def current_scope_level(self) -> int:
        """Return the current scope level (0 is global scope)"""
        return len(self.constraints_stack) - 1

    def add_constraint(self, constraint, expr):
        """Add a constraint to the current scope"""
        self.constraints_stack[-1].append((constraint, expr))

    def get_current_scope_constraints(self):
        """Get constraints in the current scope"""
        return self.constraints_stack[-1]

    def get_all_active_constraints(self):
        """Get all constraints active in current scope (including parent scopes)"""
        all_constraints = []
        for scope in self.constraints_stack:
            all_constraints.extend(scope)
        return all_constraints

    def tokenize(self, s):
        # Remove comments
        s = re.sub(';.*\n', '\n', s)
        # Add spaces around parentheses
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        # Split into tokens
        return [token for token in s.split() if token]

    def parse_tokens(self, tokens):
        if not tokens:
            return None

        if tokens[0] == '(':
            expression = []
            tokens.pop(0)  # Remove opening '('
            while tokens and tokens[0] != ')':
                exp = self.parse_tokens(tokens)
                if exp is not None:
                    expression.append(exp)
            if tokens:
                tokens.pop(0)  # Remove closing ')'
            return expression
        else:
            return tokens.pop(0)

    def create_variable(self, name, sort):
        """Create a variable of the specified sort"""
        if isinstance(sort, str):
            if sort == 'Real':
                return Real(name)
            elif sort == 'Int':
                return Int(name)
            else:
                raise ValueError(f"Unsupported sort: {sort}")
        elif isinstance(sort, list):
            # Handle BitVec sort: (_ BitVec N)
            if len(sort) == 3 and sort[0] == '_' and sort[1] == 'BitVec':
                width = int(sort[2])
                return BitVec(name, width)
            else:
                raise ValueError(f"Unsupported sort format: {sort}")
        else:
            raise ValueError(f"Invalid sort specification: {sort}")

    def process_command(self, command):
        if not isinstance(command, list):
            return

        cmd = command[0]

        if cmd == 'set-logic':
            self.logic = command[1]

        elif cmd == 'declare-const':
            name = command[1]
            # Handle different sort formats
            if isinstance(command[2], list):
                # This is for (_ BitVec N) case
                sort = command[2]
            else:
                # This is for simple sorts like Real or Int
                sort = command[2]
            self.variables[name] = self.create_variable(name, sort)

        elif cmd == 'assert':
            expr = self.build_expression(command[1])
            self.solver.add(expr)
            # Store both the original constraint and the built z3 expression
            self.add_constraint(command[1], expr)

        elif cmd == 'push':
            self.solver.push()
            # Create new scope for constraints
            self.constraints_stack.append([])

        elif cmd == 'pop':
            if len(self.constraints_stack) <= 1:
                raise ValueError("Cannot pop global scope")
            self.solver.pop()
            # Remove constraints from the current scope
            popped_constraints = self.constraints_stack.pop()
            print(f"Popped constraints from scope {len(self.constraints_stack)}:")
            for original, _ in popped_constraints:
                print(f"  {original}")

        elif cmd == 'check-sat':
            print(self.solver)
            result = self.solver.check()
            print(f"check-sat result: {result}")
            print("Current scope constraints:")
            for original, _ in self.get_current_scope_constraints():
                print(f"  {original}")

    def parse_constant(self, value, sort=None):
        """Parse a constant value based on the current logic"""
        try:
            if self.logic == 'QF_LRA':
                return float(value)
            elif self.logic == 'QF_LIA':
                return int(value)
            elif self.logic == 'QF_BV':
                if value.startswith('#b'):
                    return BitVecVal(int(value[2:], 2), len(value[2:]))
                elif value.startswith('#x'):
                    return BitVecVal(int(value[2:], 16), len(value[2:]) * 4)
                else:
                    return BitVecVal(int(value), 32)  # Default to 32 bits
            else:
                return value
        except ValueError:
            return value

    def build_expression(self, expr):
        if not isinstance(expr, list):
            if expr in self.variables:
                return self.variables[expr]
            return self.parse_constant(expr)

        op = expr[0]
        args = [self.build_expression(arg) for arg in expr[1:]]

        if op in ['+', '-', '*', '/', '>', '<', '>=', '<=', '=']:
            return self.build_arithmetic_expression(op, args)
        elif op.startswith('bv'):
            return self.build_bitvector_expression(op, args)
        else:
            raise ValueError(f"Unknown operator: {op}")

    def build_arithmetic_expression(self, op, args):
        """Build arithmetic expression based on the current logic"""
        if op == '+':
            return sum(args)
        elif op == '-':
            if len(args) == 1:
                return -args[0]
            return args[0] - args[1]
        elif op == '*':
            result = args[0]
            for arg in args[1:]:
                result *= arg
            return result
        elif op == '/':
            return args[0] / args[1]
        elif op == '>':
            return args[0] > args[1]
        elif op == '<':
            return args[0] < args[1]
        elif op == '>=':
            return args[0] >= args[1]
        elif op == '<=':
            return args[0] <= args[1]
        elif op == '=':
            return args[0] == args[1]

    def build_bitvector_expression(self, op, args):
        """Build bit-vector expression with comprehensive operation support"""
        # Comparison operations
        if op == 'bvult':
            return ULT(args[0], args[1])
        elif op == 'bvule':
            return ULE(args[0], args[1])
        elif op == 'bvugt':
            return UGT(args[0], args[1])
        elif op == 'bvuge':
            return UGE(args[0], args[1])
        elif op == 'bvslt':
            return args[0] < args[1]
        elif op == 'bvsle':
            return args[0] <= args[1]
        elif op == 'bvsgt':
            return args[0] > args[1]
        elif op == 'bvsge':
            return args[0] >= args[1]

        # Arithmetic operations
        elif op == 'bvneg':
            return -args[0]
        elif op == 'bvadd':
            return args[0] + args[1]
        elif op == 'bvsub':
            return args[0] - args[1]
        elif op == 'bvmul':
            return args[0] * args[1]
        elif op == 'bvudiv':
            return UDiv(args[0], args[1])
        elif op == 'bvsdiv':
            return args[0] / args[1]
        elif op == 'bvurem':
            return URem(args[0], args[1])
        elif op == 'bvsrem':
            return SRem(args[0], args[1])
        elif op == 'bvsmod':
            return SMod(args[0], args[1])

        # Bitwise operations
        elif op == 'bvand':
            return args[0] & args[1]
        elif op == 'bvor':
            return args[0] | args[1]
        elif op == 'bvxor':
            return args[0] ^ args[1]
        elif op == 'bvnot':
            return ~args[0]
        elif op == 'bvnand':
            return ~(args[0] & args[1])
        elif op == 'bvnor':
            return ~(args[0] | args[1])
        elif op == 'bvxnor':
            return ~(args[0] ^ args[1])

        # Shift operations
        elif op == 'bvshl':
            return args[0] << args[1]
        elif op == 'bvlshr':
            return LShR(args[0], args[1])
        elif op == 'bvashr':
            return args[0] >> args[1]

        else:
            raise ValueError(f"Unknown bit-vector operator: {op}")

    def parse_file(self, content):
        tokens = self.tokenize(content)
        while tokens:
            command = self.parse_tokens(tokens)
            if command:
                self.process_command(command)


# Example usage
smt_content_bv = """
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(assert (= (bvadd x y) #x0F))
(push)
(assert (bvult x (bvsub x y)))
(check-sat)
(pop)
(assert (bvult x y))
(check-sat)
"""

parser = SMTLIBParser()
parser.parse_file(smt_content_bv)
