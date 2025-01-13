"""
A Python program that can be called via IPC. It can take SMT-LIB2 commands
(e.g., declare-const, assert, check-sat, push/pop, get-model, get-value, etc)
from another program, and response to those commands.

There are several benefits:
- We can use other tools (e.g, pySMT) to extend the capability of Z3 (e.g., ITP, AllSMT, etc.)
- Many program analysis tools (e.g., Manticore) already interact with SMT solvers
 in this way
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

import z3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


@dataclass
class ScopeFrame:
    variables: Dict[str, Any]
    assertions: List[z3.ExprRef]


class SmtServer:
    def __init__(self, input_pipe="/tmp/smt_input", output_pipe="/tmp/smt_output"):
        self.input_pipe = input_pipe
        self.output_pipe = output_pipe
        self.solver = z3.Solver()
        self.variables: Dict[str, Any] = {}
        self.running = True
        self.scope_stack: List[ScopeFrame] = [ScopeFrame({}, [])]
        self._setup_pipes()

    def _setup_pipes(self):
        """Create named pipes with proper error handling."""
        for pipe in (self.input_pipe, self.output_pipe):
            try:
                if not os.path.exists(pipe):
                    os.mkfifo(pipe)
                    os.chmod(pipe, 0o666)  # Make pipes readable/writable by all users
                    logging.debug(f"Created FIFO: {pipe}")
            except OSError as e:
                logging.error(f"Failed to create FIFO {pipe}: {e}")
                raise

    @property
    def current_scope(self) -> ScopeFrame:
        """Get the current scope frame."""
        return self.scope_stack[-1]

    def write_response(self, response: str):
        """Write response to output pipe with error handling."""
        try:
            with open(self.output_pipe, 'w') as f:
                f.write(response + '\n')
                f.flush()
        except IOError as e:
            logging.error(f"Failed to write response: {e}")
            raise

    def parse_smt2_expr(self, expr_str: str) -> z3.ExprRef:
        """Parse SMT-LIB2 expression string into Z3 expression."""
        try:
            # Basic S-expression parsing
            expr_str = expr_str.strip()
            if not expr_str.startswith('(') or not expr_str.endswith(')'):
                # Single term (variable or constant)
                if expr_str in self.current_scope.variables:
                    return self.current_scope.variables[expr_str]
                try:
                    return z3.Int(expr_str) if expr_str.isdigit() else z3.Bool(expr_str)
                except Exception:
                    raise ValueError(f"Invalid expression: {expr_str}")

            # Remove outer parentheses and split
            inner = expr_str[1:-1].strip()
            tokens = []
            current = ''
            paren_count = 0

            # Handle nested expressions
            for char in inner:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char.isspace() and paren_count == 0:
                    if current:
                        tokens.append(current)
                        current = ''
                    continue
                current += char
            if current:
                tokens.append(current)

            if not tokens:
                raise ValueError("Empty expression")

            op = tokens[0]
            args = [self.parse_smt2_expr(t) for t in tokens[1:]]

            # Handle operations
            if op in ('and', 'or', 'not', '=', '>=', '<=', '>', '<', '+', '-', '*', '/'):
                op_map = {
                    'and': z3.And, 'or': z3.Or, 'not': z3.Not,
                    '=': lambda x, y: x == y,
                    '>=': lambda x, y: x >= y, '<=': lambda x, y: x <= y,
                    '>': lambda x, y: x > y, '<': lambda x, y: x < y,
                    '+': lambda x, y: x + y, '-': lambda x, y: x - y,
                    '*': lambda x, y: x * y, '/': lambda x, y: x / y
                }
                return op_map[op](*args)

            raise ValueError(f"Unknown operator: {op}")
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {e}")

    def handle_declare_const(self, name: str, sort: str):
        """Handle declare-const command with improved type handling."""
        try:
            if name in self.current_scope.variables:
                return "error: variable already declared"

            sort_map = {
                "Int": z3.IntSort(),
                "Bool": z3.BoolSort(),
                "Real": z3.RealSort()
            }

            if sort not in sort_map:
                return f"error: unsupported sort {sort}"

            self.current_scope.variables[name] = z3.Const(name, sort_map[sort])
            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_assert(self, expr_str: str):
        """Handle assert command with improved expression parsing."""
        try:
            expr = self.parse_smt2_expr(expr_str)
            self.solver.add(expr)
            self.current_scope.assertions.append(expr)
            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_push(self) -> str:
        """Handle push command with scope management."""
        try:
            self.solver.push()
            # Create new scope with copies of current variables
            new_scope = ScopeFrame(dict(self.current_scope.variables), [])
            self.scope_stack.append(new_scope)
            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_pop(self) -> str:
        """Handle pop command with scope management."""
        try:
            if len(self.scope_stack) <= 1:
                return "error: cannot pop the global scope"
            self.solver.pop()
            self.scope_stack.pop()
            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_command(self, command: str) -> str:
        """Handle SMT-LIB2 commands with improved parsing."""
        try:
            parts = command.strip().split(maxsplit=1)
            if not parts:
                return ""

            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            handlers = {
                "exit": lambda _: self._handle_exit(),
                "declare-const": lambda a: self._handle_declare_const_args(a),
                "assert": lambda a: self.handle_assert(a),
                "check-sat": lambda _: str(self.solver.check()),
                "get-model": lambda _: self._handle_get_model(),
                "push": lambda _: self.handle_push(),
                "pop": lambda _: self.handle_pop(),
                "get-value": lambda a: self._handle_get_value(a)
            }

            if cmd not in handlers:
                return f"error: unknown command {cmd}"

            return handlers[cmd](args)
        except Exception as e:
            return f"error: {str(e)}"

    def _handle_exit(self) -> str:
        self.running = False
        return "bye"

    def _handle_declare_const_args(self, args: str) -> str:
        parts = args.split()
        if len(parts) != 2:
            return "error: declare-const requires name and sort"
        return self.handle_declare_const(parts[0], parts[1])

    def _handle_get_model(self) -> str:
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            return " ".join(f"({k} {model[v]})" for k, v in self.current_scope.variables.items())
        return "unknown"

    def _handle_get_value(self, args: str) -> str:
        if self.solver.check() != z3.sat:
            return "unknown"
        try:
            model = self.solver.model()
            values = []
            for var_name in args.split():
                if var_name in self.current_scope.variables:
                    val = model.evaluate(self.current_scope.variables[var_name])
                    values.append(f"({var_name} {val})")
            return "(" + " ".join(values) + ")"
        except Exception as e:
            return f"error: {str(e)}"

    def run(self):
        """Run the server with improved error handling."""
        while self.running:
            try:
                with open(self.input_pipe, 'r') as f:
                    for line in f:
                        command = line.strip()
                        if command:
                            response = self.handle_command(command)
                            self.write_response(response)
                        if not self.running:
                            break
            except IOError as e:
                logging.error(f"IO Error: {e}")
                if not self.running:
                    break
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                self.write_response(f"error: {str(e)}")


if __name__ == "__main__":
    try:
        server = SmtServer()
        server.run()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
