"""Helper methods for Symbolic Abstraction analysis of straight-line programs.
"""
import z3
from ..domains.algorithms import bilateral


class Program:
    """Represents a straight-line program in 2-operand notation.
    """

    def __init__(self, program):
        """Initialize the program from a string @program.

        See example_program.py for an example of the two-operand notation we
        support.
        """
        program = [
            line.strip() for line in program.split("\n") if line.strip()]
        program = [line.split(" ") for line in program]

        # (inout, op, in)
        assert all(len(statement) == 3 for statement in program)

        inouts, _ops, ins = zip(*program)
        self.variables = set(inouts) | set(ins)
        self.variables = sorted(variable for variable in self.variables
                                if not variable.isdigit())
        assert all("'" not in variable for variable in self.variables)

        z3_variables = [list(map(z3.Int, self.variables))]
        z3_statements = []
        for i, (operand_to, operation, operand_from) in enumerate(program):
            primed = [variable + ((i + 1) * "'") for variable in self.variables]
            z3_variables.append(list(map(z3.Int, primed)))

            pre_variables = z3_variables[-2]
            post_variables = z3_variables[-1]

            try:
                operand_from_index = self.variables.index(operand_from)
                pre_operand_from = pre_variables[operand_from_index]
            except ValueError:
                pre_operand_from = int(operand_from)

            operand_to_index = self.variables.index(operand_to)
            pre_operand_to = pre_variables[operand_to_index]
            post_operand_to = post_variables[operand_to_index]

            if operation == "=":
                z3_statements.append(post_operand_to == pre_operand_from)
            elif operation == "<=":
                z3_statements.append(post_operand_to <= pre_operand_from)
            elif operation == ">=":
                z3_statements.append(post_operand_to >= pre_operand_from)
            elif operation == "+=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to + pre_operand_from))
            elif operation == "-=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to - pre_operand_from))
            elif operation == "*=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to * pre_operand_from))
            elif operation == "/=":
                z3_statements.append(
                    post_operand_to == (pre_operand_to / pre_operand_from))
            else:
                raise NotImplementedError

            for pre, post in zip(pre_variables, post_variables):
                if post is not post_operand_to:
                    z3_statements.append(pre == post)

        self.prime_depth = len(program)
        self.z3_formula = z3.And(*z3_statements)

    def transform(self, domain, input_abstract_state):
        """Compute the most precise output abstract state.
        """

        def add_primes(unprimed):
            return unprimed + (self.prime_depth * "'")

        output_domain = domain.translate(dict({
            unprimed: add_primes(unprimed) for unprimed in domain.variables
        }))

        phi = z3.And(self.z3_formula, domain.gamma_hat(input_abstract_state))
        output_state = bilateral(output_domain, phi)

        return output_state.translate(dict({
            add_primes(unprimed): unprimed for unprimed in domain.variables
        }))
