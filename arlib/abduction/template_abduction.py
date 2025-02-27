"""Template-based abductive inference.

Infer abductive hypotheissis expressible via a given template,
e.g., the elements in an interval domain, the Boolean combination of a given set of predicates, etc.?

For example:
  Let x, y be the variables used in the abductive hypothesis and the template is the interval domain. 
  We can use the template: (x \in [a, b]) and (y \in [c, d])
  to infer the abductive hypothesis.

  When the algorithm is complete, we can make conclusive arguments: 
  If it fails, there is no such abductive hypothesis (expressible in the interval domain).

TBD: check for correctness and more tests.
"""

from typing import List, Callable, Any, Set, Optional, Tuple
import z3


class TemplateAbduction:
    def __init__(self, variables: List[z3.ExprRef], template_type: str):
        self.variables = variables
        self.template_type = template_type
        self.solver = z3.Solver()
        
    def create_interval_template(self) -> z3.ExprRef:
        """Create an interval template for each variable."""
        constraints = []
        for var in self.variables:
            # Create bounds for each variable
            lower = z3.Real(f'lower_{var}')
            upper = z3.Real(f'upper_{var}')
            # Add interval constraints
            constraints.append(z3.And(var >= lower, var <= upper))
        return z3.And(constraints)

    def abduce(self, precond: z3.ExprRef, postcond: z3.ExprRef) -> Optional[z3.ExprRef]:
        """
        Perform template-based abduction.
        
        Args:
            precond: Precondition Γ
            postcond: Postcondition φ
            
        Returns:
            The abductive hypothesis ψ if found, None otherwise
        """
        # Create template based on type
        if self.template_type == "interval":
            template = self.create_interval_template()
        else:
            raise ValueError(f"Unsupported template type: {self.template_type}")

        # Add consistency constraint: Γ ∧ ψ is satisfiable
        self.solver.push()
        self.solver.add(precond)
        self.solver.add(template)
        
        if self.solver.check() == z3.unsat:
            self.solver.pop()
            return None
        
        self.solver.pop()

        # Add entailment constraint: Γ ∧ ψ |= φ
        self.solver.push()
        self.solver.add(z3.Not(z3.Implies(z3.And(precond, template), postcond)))
        
        if self.solver.check() == z3.sat:
            self.solver.pop()
            return None
            
        self.solver.pop()

        # Extract solution
        self.solver.push()
        self.solver.add(template)
        
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            result = self.instantiate_template(template, model)
            self.solver.pop()
            return result
            
        self.solver.pop()
        return None

    def instantiate_template(self, template: z3.ExprRef, model: z3.ModelRef) -> z3.ExprRef:
        """Instantiate template with concrete values from model."""
        substitutions = []
        for var in self.variables:
            lower = model.eval(z3.Real(f'lower_{var}'))
            upper = model.eval(z3.Real(f'upper_{var}'))
            substitutions.append((var >= lower) & (var <= upper))
        return z3.And(substitutions)


def test_interval_abduction():
    """Test case for interval-based abduction."""
    x, y = z3.Reals('x y')
    
    # Create abduction instance
    abductor = TemplateAbduction([x, y], "interval")
    
    # Test case: x > 0 |- y > x
    precond = x > 0
    postcond = y > x
    
    result = abductor.abduce(precond, postcond)
    
    if result is not None:
        print("Found abductive hypothesis:")
        print(result)
    else:
        print("No valid abductive hypothesis found")


if __name__ == "__main__":
    test_interval_abduction()
