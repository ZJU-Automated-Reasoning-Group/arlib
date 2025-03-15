"""
Z3-based AllSMT solver implementation.

This module provides an implementation of the AllSMT solver using Z3.
"""

from typing import List, Any, Dict, Optional
from z3 import Solver, sat, Or

from .base import AllSMTSolver


class Z3AllSMTSolver(AllSMTSolver):
    """
    Z3-based AllSMT solver implementation.
    
    This class implements the AllSMT solver interface using Z3 as the underlying solver.
    """
    
    def __init__(self):
        """Initialize the Z3-based AllSMT solver."""
        self._models = []
        self._model_count = 0
        self._model_limit_reached = False
    
    def solve(self, expr, keys, model_limit: int = 100):
        """
        Enumerate all satisfying models for the given expression over the specified keys.
        
        Args:
            expr: The Z3 expression/formula to solve
            keys: The Z3 variables to track in the models
            model_limit: Maximum number of models to generate (default: 100)
            
        Returns:
            List of Z3 models satisfying the expression
        """
        solver = Solver()
        solver.add(expr)
        self._models = []
        self._model_count = 0
        self._model_limit_reached = False
        
        while solver.check() == sat:
            model = solver.model()
            self._model_count += 1
            self._models.append(model)
            
            # Check if we've reached the model limit
            if self._model_count >= model_limit:
                self._model_limit_reached = True
                break
            
            # Create blocking clause to exclude the current model
            block = []
            for k in keys:
                block.append(k != model[k])
            solver.add(Or(block))
        
        return self._models
    
    def get_model_count(self) -> int:
        """
        Get the number of models found in the last solve call.
        
        Returns:
            int: The number of models
        """
        return self._model_count
    
    @property
    def models(self):
        """
        Get all models found in the last solve call.
        
        Returns:
            List of Z3 models
        """
        return self._models
    
    def print_models(self, verbose: bool = False):
        """
        Print all models found in the last solve call.
        
        Args:
            verbose: Whether to print detailed information about each model
        """
        if not self._models:
            print("No models found.")
            return
        
        for i, model in enumerate(self._models):
            if verbose:
                print(f"Model {i+1}:")
                for decl in model.decls():
                    print(f"  {decl.name()} = {model[decl]}")
            else:
                print(f"Model {i+1}: {model}")
        
        if self._model_limit_reached:
            print(f"Model limit reached. Found {self._model_count} models (there may be more).")
        else:
            print(f"Total number of models: {self._model_count}")


def demo():
    """Demonstrate the usage of the Z3-based AllSMT solver."""
    from z3 import Ints, And
    
    x, y = Ints('x y')
    expr = And(x + y == 5, x > 0, y > 0)
    
    solver = Z3AllSMTSolver()
    solver.solve(expr, [x, y], model_limit=10)
    solver.print_models(verbose=True)


if __name__ == "__main__":
    demo() 