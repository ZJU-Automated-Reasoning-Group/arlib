"""
Base class and common utilities for MaxSMT solvers.

Provides the abstract base class for MaxSMT solvers and common utility functions.
"""

import logging
from typing import List, Tuple, Optional, Set, Any
from enum import Enum

import z3

logger = logging.getLogger(__name__)


class MaxSMTAlgorithm(Enum):
    """Enumeration of available MaxSMT algorithms"""
    CORE_GUIDED = "core-guided"   # Core-guided algorithm from SAT'13
    IHS = "ihs"                   # Implicit Hitting Set algorithm from IJCAR'18
    LOCAL_SEARCH = "local-search"  # Local search for MaxSMT(LIA)


class MaxSMTSolverBase:
    """
    Base class for MaxSMT solvers
    """
    def __init__(self, solver_name: str = "z3"):
        """Initialize the MaxSMT solver

        Args:
            solver_name: Name of the underlying SMT solver
        """
        self.solver_name = solver_name
        self.hard_constraints = []
        self.soft_constraints = []
        self.weights = []
        self.solver = None
        self._setup_solver(solver_name)

    def _setup_solver(self, solver_name: str):
        """Set up the underlying SMT solver
        
        Args:
            solver_name: Name of the SMT solver to use
        """
        if solver_name == "z3":
            self.solver = z3.Solver()
        else:
            # For other solvers, we could add adapter code here
            raise NotImplementedError(f"Solver {solver_name} not supported")

    def add_hard_constraint(self, constraint):
        """Add a hard constraint (must be satisfied)
        
        Args:
            constraint: SMT formula that must be satisfied
        """
        self.hard_constraints.append(constraint)
    
    def add_hard_constraints(self, constraints):
        """Add multiple hard constraints
        
        Args:
            constraints: List of SMT formulas that must be satisfied
        """
        self.hard_constraints.extend(constraints)

    def add_soft_constraint(self, constraint, weight: float = 1.0):
        """Add a soft constraint with a weight
        
        Args:
            constraint: SMT formula that should be satisfied if possible
            weight: Weight of the constraint (higher = more important)
        """
        self.soft_constraints.append(constraint)
        self.weights.append(weight)
    
    def add_soft_constraints(self, constraints, weights=None):
        """Add multiple soft constraints with weights
        
        Args:
            constraints: List of SMT formulas 
            weights: List of weights (default: all 1.0)
        """
        if weights is None:
            weights = [1.0] * len(constraints)
        
        if len(constraints) != len(weights):
            raise ValueError("Number of constraints must match number of weights")
        
        self.soft_constraints.extend(constraints)
        self.weights.extend(weights)
    
    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Solve the MaxSMT problem
        
        Returns:
            Tuple of (sat, model, optimal_cost)
            sat: True if a solution was found
            model: z3 model if sat is True, None otherwise
            optimal_cost: Sum of weights of unsatisfied soft constraints
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_variables(self, formula) -> Set[Any]:
        """Extract variables from a Z3 formula"""
        variables = set()
        
        def collect(expr):
            if z3.is_const(expr) and not z3.is_bool(expr) and not z3.is_true(expr) and not z3.is_false(expr):
                variables.add(expr)
            else:
                for child in expr.children():
                    collect(child)
        
        collect(formula)
        return variables
    
    def _calculate_cost(self, model) -> float:
        """Calculate the cost of a model (sum of weights of unsatisfied soft constraints)"""
        cost = 0.0
        for i, sc in enumerate(self.soft_constraints):
            if not self._evaluate(model, sc):
                cost += self.weights[i]
        return cost
    
    def _evaluate(self, model, formula) -> bool:
        """Evaluate a formula under a model"""
        try:
            return z3.is_true(model.eval(formula, model_completion=True))
        except:
            # For complex formulas that can't be directly evaluated
            return False
    
    def _satisfies_hard_constraints(self, model) -> bool:
        """Check if a model satisfies all hard constraints"""
        for hc in self.hard_constraints:
            if not self._evaluate(model, hc):
                return False
        return True 