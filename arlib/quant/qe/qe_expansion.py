"""
Quantifier Elimination for Boolean formulas via Shannon Expansion

(FIXME: this file is generated by LLM)
"""

from pysat.formula import CNF
from pysat.solvers import Glucose3


class QuantifierElimination:
    def __init__(self):
        self.var_counter = 1
        self.var_map = {}
        self.reverse_map = {}

    def get_var_id(self, var_name):
        """Map string variable names to integer IDs used by PySAT."""
        if var_name not in self.var_map:
            self.var_map[var_name] = self.var_counter
            self.reverse_map[self.var_counter] = var_name
            self.var_counter += 1
        return self.var_map[var_name]

    def shannon_expand(self, formula, var_id):
        """Perform Shannon expansion using PySAT."""
        # Create positive cofactor (var = True)
        pos_cofactor = CNF()
        for clause in formula.clauses:
            new_clause = []
            for lit in clause:
                if abs(lit) != var_id:
                    new_clause.append(lit)
                elif lit > 0:
                    # Variable appears positively, clause is satisfied
                    new_clause = None
                    break
            if new_clause is not None:
                if new_clause:  # Only add non-empty clauses
                    pos_cofactor.append(new_clause)

        # Create negative cofactor (var = False)
        neg_cofactor = CNF()
        for clause in formula.clauses:
            new_clause = []
            for lit in clause:
                if abs(lit) != var_id:
                    new_clause.append(lit)
                elif lit < 0:
                    # Variable appears negatively, clause is satisfied
                    new_clause = None
                    break
            if new_clause is not None:
                if new_clause:  # Only add non-empty clauses
                    neg_cofactor.append(new_clause)

        return pos_cofactor, neg_cofactor

    def eliminate_exists(self, formula, var_id):
        """Eliminate existential quantifier using Shannon expansion."""
        pos_cofactor, neg_cofactor = self.shannon_expand(formula, var_id)

        # Combine cofactors with OR
        result = CNF()

        # If either cofactor is empty (valid), the result is valid
        if not pos_cofactor.clauses or not neg_cofactor.clauses:
            return CNF()

        # Otherwise, take the conjunction of all resolvents
        for clause1 in pos_cofactor.clauses:
            for clause2 in neg_cofactor.clauses:
                resolvent = self.resolve_clauses(clause1, clause2)
                if resolvent:  # Only add non-empty resolvents
                    result.append(resolvent)

        return result

    def resolve_clauses(self, clause1, clause2):
        """Compute resolvent of two clauses."""
        result = list(set(clause1 + clause2))  # Remove duplicates
        return result if result else None

    def eliminate_quantifiers(self, formula, variables):
        """Eliminate a sequence of existential quantifiers."""
        result = formula
        for var in variables:
            var_id = self.get_var_id(var)
            result = self.eliminate_exists(result, var_id)
        return result

    def is_satisfiable(self, formula):
        """Check if a CNF formula is satisfiable using PySAT."""
        with Glucose3() as solver:
            solver.append_formula(formula)
            return solver.solve()


# Example usage
if __name__ == "__main__":
    # Create a QE solver instance
    qe = QuantifierElimination()

    # Example 1: ∃x.(x ∨ y)
    formula = CNF()
    x_id = qe.get_var_id('x')
    y_id = qe.get_var_id('y')
    formula.append([x_id, y_id])  # Clause: x ∨ y

    print("Example 1: ∃x.(x ∨ y)")
    print("Original formula:", formula.clauses)
    result = qe.eliminate_quantifiers(formula, ['x'])
    print("After eliminating x:", result.clauses)
    print("Satisfiable:", qe.is_satisfiable(result))

    # Example 2: ∃x.(x ∨ y) ∧ (¬x ∨ z)
    formula = CNF()
    x_id = qe.get_var_id('x')
    y_id = qe.get_var_id('y')
    z_id = qe.get_var_id('z')
    formula.append([x_id, y_id])  # Clause: x ∨ y
    formula.append([-x_id, z_id])  # Clause: ¬x ∨ z

    print("\nExample 2: ∃x.(x ∨ y) ∧ (¬x ∨ z)")
    print("Original formula:", formula.clauses)
    result = qe.eliminate_quantifiers(formula, ['x'])
    print("After eliminating x:", result.clauses)
    print("Satisfiable:", qe.is_satisfiable(result))

    # Example 3: ∃x.(x) ∧ (¬x)
    formula = CNF()
    x_id = qe.get_var_id('x')
    formula.append([x_id])  # Clause: x
    formula.append([-x_id])  # Clause: ¬x

    print("\nExample 3: ∃x.(x) ∧ (¬x)")
    print("Original formula:", formula.clauses)
    result = qe.eliminate_quantifiers(formula, ['x'])
    print("After eliminating x:", result.clauses)
    print("Satisfiable:", qe.is_satisfiable(result))