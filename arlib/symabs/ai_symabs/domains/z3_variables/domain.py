"""Base class definitions for conjunctive domains modelable by Z3.
"""

import z3
from ..core import ConjunctiveDomain
from .concrete import Z3VariablesState


# pylint: disable=abstract-method
class Z3VariablesDomain(ConjunctiveDomain):
    """Represents an abstract space modelable by Z3.
    """

    def __init__(self, variables, variable_type=z3.Int,
                 build_z3_variables=True):
        """Constructs a new Z3VariablesDomain, with variables @variables.

        Arguments
        =========
        - @variables should be a list of variable names.
        - @variable_type is the z3 variable type that should be used.
        """
        self.variables = variables
        self.variable_type = variable_type
        if build_z3_variables:
            self.z3_variables = dict((name, variable_type(name))
                                     for name in self.variables)
        else:
            self.z3_variables = None
        self.concrete_type = Z3VariablesState
        self.iterative_solvers = {}

    def z3_variable(self, name):
        """Returns the Z3 variable associated with name.
        """
        return self.z3_variables[name]

    def model(self, phi):
        """Returns a solution to phi.

        If none exists (i.e. phi is unsatisfiable), None is returned.
        """
        solver = z3.Solver()
        solver.add(phi)
        if solver.check() == z3.sat:
            model = solver.model()
            if self.variable_type == z3.Int:
                solution = dict((d.name(), model.eval(d()).as_long())
                                for d in model.decls())
            else:
                solution = dict((d.name(), model.eval(d()).as_fraction())
                                for d in model.decls())
            for name in self.variables:
                if name not in solution:
                    solution[name] = 0
            return Z3VariablesState(solution, self.variable_type)
        return None

    def model_and(self, phi1, phi2):
        """Returns a solution to phi and phi2.

        If none exists (i.e. phi is unsatisfiable), None is returned. This
        method will use the iterative solver as long as phi1 has been seen
        before.
        """
        if id(phi1) not in self.iterative_solvers:
            self.iterative_solvers[id(phi1)] = z3.Solver()
            self.iterative_solvers[id(phi1)].add(phi1)
        solver = self.iterative_solvers[id(phi1)]
        solver.push()
        solver.add(phi2)
        if solver.check() == z3.sat:
            model = solver.model()
            if self.variable_type == z3.Int:
                solution = dict((d.name(), model.eval(d()).as_long())
                                for d in model.decls())
            else:
                solution = dict((d.name(), model.eval(d()).as_fraction())
                                for d in model.decls())
            for name in self.variables:
                if name not in solution:
                    solution[name] = 0
            solver.pop()
            return Z3VariablesState(solution, self.variable_type)
        solver.pop()
        return None

    def logic_and(self, formulas):
        """Returns the logical and of the given formulas.
        """
        return z3.And(formulas)

    def logic_not(self, formula):
        """Returns the logical negation of the given formula.
        """
        return z3.Not(formula)

    def translate(self, translation):
        return type(self)(list(map(translation.__getitem__, self.variables)))
