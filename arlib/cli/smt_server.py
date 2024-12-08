"""
A Python program that can be called via IPC. It can take SMT-LIB2 commands
(e.g., delcare-const, assert, check-sat, push/pop, get-model, get-value, etc)
from another program, and response to those commands.
"""
import os
from typing import Dict, Any

import z3


class SmtServer:
    def __init__(self, input_pipe="/tmp/smt_input", output_pipe="/tmp/smt_output"):
        self.input_pipe = input_pipe
        self.output_pipe = output_pipe
        self.solver = z3.Solver()
        self.variables: Dict[str, Any] = {}
        self.running = True

        # Create pipes if they don't exist
        for pipe in (input_pipe, output_pipe):
            if not os.path.exists(pipe):
                os.mkfifo(pipe)

    def write_response(self, response: str):
        with open(self.output_pipe, 'w') as f:
            f.write(response + '\n')
            f.flush()

    def handle_declare_const(self, name: str, sort: str):
        if sort.lower() == "int":
            self.variables[name] = z3.Int(name)
        elif sort.lower() == "bool":
            self.variables[name] = z3.Bool(name)
        elif sort.lower() == "real":
            self.variables[name] = z3.Real(name)
        else:
            return "unsupported"
        return "success"

    def handle_assert(self, expr_str: str):
        try:
            # Replace variable names with their Z3 objects
            for name, var in self.variables.items():
                expr_str = expr_str.replace(name, f"self.variables['{name}']")

            # Evaluate the expression in the context of this object
            expr = eval(expr_str)
            self.solver.add(expr)
            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def handle_command(self, command: str) -> str:
        parts = command.strip().split()
        if not parts:
            return ""

        cmd = parts[0].lower()

        if cmd == "exit":
            self.running = False
            return "bye"

        elif cmd == "declare-const":
            if len(parts) != 3:
                return "error: declare-const requires name and sort"
            return self.handle_declare_const(parts[1], parts[2])

        elif cmd == "assert":
            if len(parts) < 2:
                return "error: assert requires an expression"
            return self.handle_assert(" ".join(parts[1:]))

        elif cmd == "check-sat":
            result = self.solver.check()
            return str(result)

        elif cmd == "get-model":
            if self.solver.check() == z3.sat:
                return str(self.solver.model())
            return "unknown"

        elif cmd == "push":
            self.solver.push()
            return "success"

        elif cmd == "pop":
            self.solver.pop()
            return "success"

        elif cmd == "get-value":
            if len(parts) < 2:
                return "error: get-value requires variable names"
            if self.solver.check() != z3.sat:
                return "unknown"
            model = self.solver.model()
            values = []
            for var_name in parts[1:]:
                if var_name in self.variables:
                    values.append(f"{var_name}={model.evaluate(self.variables[var_name])}")
            return " ".join(values)

        return f"error: unknown command {cmd}"

    def run(self):
        while self.running:
            try:
                with open(self.input_pipe, 'r') as f:
                    while self.running:
                        command = f.readline().strip()
                        if command:
                            response = self.handle_command(command)
                            self.write_response(response)
            except Exception as e:
                self.write_response(f"error: {str(e)}")


if __name__ == "__main__":
    server = SmtServer()
    server.run()
