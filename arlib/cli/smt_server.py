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
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Optional, Callable, Union

import z3

# Import advanced arlib features
try:
    from arlib.allsmt import create_allsmt_solver
    from arlib.unsat_core.unsat_core import get_unsat_core, enumerate_all_mus, Algorithm as UnsatAlgorithm
    from arlib.backbone.smt_backbone_literals import get_backbone_literals
    from arlib.counting import model_counter

    ARLIB_FEATURES_AVAILABLE = True
except ImportError:
    ARLIB_FEATURES_AVAILABLE = False
    logging.warning("Some arlib features are not available. Running with limited functionality.")

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

        # Advanced features configuration
        self.allsmt_model_limit = 100  # Default limit for AllSMT models
        self.unsat_core_algorithm = "marco"  # Default UNSAT core algorithm
        self.unsat_core_timeout = None  # Default timeout for UNSAT core computation
        self.model_count_timeout = 60  # Default timeout for model counting (in seconds)

    def _setup_pipes(self):
        """Create named pipes with proper error handling."""
        for pipe in (self.input_pipe, self.output_pipe):
            try:
                # Remove pipe if it already exists (might be stale)
                if os.path.exists(pipe):
                    try:
                        os.unlink(pipe)
                        logging.debug(f"Removed existing FIFO: {pipe}")
                    except OSError as e:
                        logging.warning(f"Failed to remove existing FIFO {pipe}: {e}")

                # Create new pipe
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

    def handle_allsmt(self, args: str) -> str:
        """Handle allsmt command to enumerate all satisfying models."""
        if not ARLIB_FEATURES_AVAILABLE:
            return "error: allsmt feature not available"

        try:
            # Parse arguments
            parts = args.split()
            if not parts:
                return "error: allsmt requires variable names"

            # Extract model limit if provided
            model_limit = self.allsmt_model_limit
            if parts[0].startswith(":limit="):
                limit_str = parts[0].split("=")[1]
                try:
                    model_limit = int(limit_str)
                    parts = parts[1:]
                except ValueError:
                    return f"error: invalid model limit: {limit_str}"

            # Get variables to project onto
            var_names = parts
            variables = []
            for name in var_names:
                if name not in self.current_scope.variables:
                    return f"error: variable {name} not declared"
                variables.append(self.current_scope.variables[name])

            # Create formula from current assertions
            formula = z3.And(*self.current_scope.assertions) if self.current_scope.assertions else z3.BoolVal(True)

            # Create AllSMT solver and enumerate models
            solver = create_allsmt_solver()
            models = solver.get_all_sat(formula, variables, model_limit=model_limit)

            # Format results
            result = []
            for model in models:
                model_str = []
                for var_name, var in zip(var_names, variables):
                    value = model.get(var)
                    model_str.append(f"({var_name} {value})")
                result.append("(" + " ".join(model_str) + ")")

            return "(" + " ".join(result) + ")"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_unsat_core(self, args: str) -> str:
        """Handle unsat-core command to compute unsatisfiable cores."""
        if not ARLIB_FEATURES_AVAILABLE:
            return "error: unsat-core feature not available"

        try:
            # Parse arguments
            parts = args.split()
            algorithm = self.unsat_core_algorithm
            timeout = self.unsat_core_timeout
            enumerate_all = False

            # Process options
            i = 0
            while i < len(parts):
                if parts[i].startswith(":algorithm="):
                    algorithm = parts[i].split("=")[1]
                    parts.pop(i)
                elif parts[i].startswith(":timeout="):
                    timeout_str = parts[i].split("=")[1]
                    try:
                        timeout = int(timeout_str)
                    except ValueError:
                        return f"error: invalid timeout: {timeout_str}"
                    parts.pop(i)
                elif parts[i] == ":enumerate-all":
                    enumerate_all = True
                    parts.pop(i)
                else:
                    i += 1

            # Get assertions as constraints
            constraints = self.current_scope.assertions
            if not constraints:
                return "error: no assertions to analyze"

            # Create solver factory
            def solver_factory():
                return z3.Solver()

            # Compute unsat core
            if enumerate_all:
                result = enumerate_all_mus(constraints, solver_factory, timeout=timeout)
            else:
                result = get_unsat_core(constraints, solver_factory, algorithm=algorithm, timeout=timeout)

            # Format results
            cores = result.cores
            formatted_cores = []
            for core in cores:
                core_exprs = [constraints[idx] for idx in core]
                core_str = " ".join(str(expr) for expr in core_exprs)
                formatted_cores.append(f"({core_str})")

            return "(" + " ".join(formatted_cores) + ")"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_backbone(self, args: str) -> str:
        """Handle backbone command to compute backbone literals."""
        if not ARLIB_FEATURES_AVAILABLE:
            return "error: backbone feature not available"

        try:
            # Parse arguments
            parts = args.split()
            algorithm = "model-enumeration"  # Default algorithm

            # Process options
            if parts and parts[0].startswith(":algorithm="):
                algorithm = parts[0].split("=")[1]
                parts = parts[1:]

            # Get formula from current assertions
            formula = z3.And(*self.current_scope.assertions) if self.current_scope.assertions else z3.BoolVal(True)

            # Get all variables as potential literals
            literals = []
            for var in self.current_scope.variables.values():
                if var.sort() == z3.BoolSort():
                    literals.append(var)
                    literals.append(z3.Not(var))

            # Compute backbone literals
            backbone_lits = get_backbone_literals(formula, literals, algorithm)

            # Format results
            result = " ".join(str(lit) for lit in backbone_lits)
            return f"({result})"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_count_models(self, args: str) -> str:
        """Handle count-models command to count satisfying models."""
        if not ARLIB_FEATURES_AVAILABLE:
            return "error: model-counting feature not available"

        try:
            # Parse arguments
            parts = args.split()
            timeout = self.model_count_timeout
            approximate = False

            # Process options
            i = 0
            while i < len(parts):
                if parts[i].startswith(":timeout="):
                    timeout_str = parts[i].split("=")[1]
                    try:
                        timeout = int(timeout_str)
                    except ValueError:
                        return f"error: invalid timeout: {timeout_str}"
                    parts.pop(i)
                elif parts[i] == ":approximate":
                    approximate = True
                    parts.pop(i)
                else:
                    i += 1

            # Get formula from current assertions
            formula = z3.And(*self.current_scope.assertions) if self.current_scope.assertions else z3.BoolVal(True)

            # Count models
            if approximate:
                count = model_counter.approximate_model_count(formula, timeout=timeout)
                return f"(approximate {count})"
            else:
                count = model_counter.exact_model_count(formula, timeout=timeout)
                return f"(exact {count})"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_set_option(self, args: str) -> str:
        """Handle set-option command to configure server options."""
        try:
            parts = args.split(maxsplit=1)
            if len(parts) != 2:
                return "error: set-option requires option name and value"

            option, value = parts

            # Handle different options
            if option == ":allsmt-model-limit":
                try:
                    self.allsmt_model_limit = int(value)
                    return "success"
                except ValueError:
                    return f"error: invalid model limit: {value}"
            elif option == ":unsat-core-algorithm":
                self.unsat_core_algorithm = value
                return "success"
            elif option == ":unsat-core-timeout":
                try:
                    self.unsat_core_timeout = int(value) if value != "none" else None
                    return "success"
                except ValueError:
                    return f"error: invalid timeout: {value}"
            elif option == ":model-count-timeout":
                try:
                    self.model_count_timeout = int(value)
                    return "success"
                except ValueError:
                    return f"error: invalid timeout: {value}"
            else:
                return f"error: unknown option: {option}"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_help(self) -> str:
        """Handle help command to show available commands."""
        basic_commands = [
            "declare-const <name> <sort>",
            "assert <expr>",
            "check-sat",
            "get-model",
            "get-value <var1> <var2> ...",
            "push",
            "pop",
            "exit"
        ]

        advanced_commands = [
            "allsmt [:limit=<n>] <var1> <var2> ...",
            "unsat-core [:algorithm=<alg>] [:timeout=<n>] [:enumerate-all]",
            "backbone [:algorithm=<alg>]",
            "count-models [:timeout=<n>] [:approximate]",
            "set-option <option> <value>",
            "help"
        ]

        options = [
            ":allsmt-model-limit <n>",
            ":unsat-core-algorithm <marco|musx|optux>",
            ":unsat-core-timeout <n|none>",
            ":model-count-timeout <n>"
        ]

        result = "Available commands:\n"
        result += "Basic commands:\n  " + "\n  ".join(basic_commands) + "\n"
        result += "Advanced commands:\n  " + "\n  ".join(advanced_commands) + "\n"
        result += "Options:\n  " + "\n  ".join(options)

        return result

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
                "get-value": lambda a: self._handle_get_value(a),
                "allsmt": lambda a: self.handle_allsmt(a),
                "unsat-core": lambda a: self.handle_unsat_core(a),
                "backbone": lambda a: self.handle_backbone(a),
                "count-models": lambda a: self.handle_count_models(a),
                "set-option": lambda a: self.handle_set_option(a),
                "help": lambda _: self.handle_help()
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
        logging.info(f"SMT server started. Input pipe: {self.input_pipe}, Output pipe: {self.output_pipe}")
        logging.info("Waiting for commands...")

        while self.running:
            try:
                # Read one command at a time instead of blocking in a loop
                with open(self.input_pipe, 'r') as f:
                    line = f.readline()
                    if line:
                        command = line.strip()
                        if command:
                            logging.debug(f"Received command: {command}")
                            response = self.handle_command(command)
                            logging.debug(f"Sending response: {response}")
                            self.write_response(response)
                        if not self.running:
                            break
                    else:
                        # No input available, sleep briefly and continue
                        time.sleep(0.01)
            except IOError as e:
                logging.error(f"IO Error: {e}")
                if not self.running:
                    break
                time.sleep(0.1)
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                self.write_response(f"error: {str(e)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced SMT Server with advanced arlib features")
    parser.add_argument("--input-pipe", default="/tmp/smt_input", help="Path to input pipe (default: /tmp/smt_input)")
    parser.add_argument("--output-pipe", default="/tmp/smt_output",
                        help="Path to output pipe (default: /tmp/smt_output)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Start server
        server = SmtServer(input_pipe=args.input_pipe, output_pipe=args.output_pipe)
        server.run()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
