"""
Linear Integer and Real Arithmetic sampler implementation.

This module provides a sampler for linear integer and real arithmetic formulas.
"""

import z3
from typing import Set, Dict, Any, List
import random

from arlib.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from arlib.sampling.utils import get_vars, is_int, is_real


class LIRASampler(Sampler):
    """
    Sampler for linear integer and real arithmetic formulas.
    
    This class implements a sampler for linear integer and real arithmetic formulas using Z3.
    """

    def __init__(self, **kwargs):
        """Initialize the LIRA sampler."""
        self.formula = None
        self.variables = []

    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.
        
        Args:
            logic: The logic to check
            
        Returns:
            True if the sampler supports the logic, False otherwise
        """
        return logic in [Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA]

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """
        Initialize the sampler with a formula.
        
        Args:
            formula: The Z3 formula to sample from
        """
        self.formula = formula

        # Extract variables from the formula
        self.variables = []
        for var in get_vars(formula):
            if is_int(var) or is_real(var):
                self.variables.append(var)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        Generate samples according to the given options.
        
        Args:
            options: The sampling options
            
        Returns:
            A SamplingResult containing the generated samples
        """
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        # Set random seed if provided
        if options.random_seed is not None:
            random.seed(options.random_seed)

        # Create a solver
        solver = z3.Solver()
        solver.add(self.formula)

        # Generate samples
        samples = []
        stats = {"time_ms": 0, "iterations": 0}

        for _ in range(options.num_samples):
            if solver.check() == z3.sat:
                model = solver.model()

                # Convert model to a dictionary
                sample = {}
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    if is_int(var):
                        sample[str(var)] = value.as_long()
                    else:  # Real
                        # Convert to float using as_decimal
                        try:
                            sample[str(var)] = float(value.as_decimal(10))
                        except:
                            # Fallback to string conversion
                            sample[str(var)] = float(str(value))

                samples.append(sample)

                # Add blocking clause to prevent the same model
                # For reals, we need to be careful about exact equality
                block = []
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    if is_int(var):
                        block.append(var != value)
                    else:  # Real
                        # For reals, we add a small delta to avoid numerical issues
                        delta = 0.001
                        block.append(z3.Or(var < value - delta, var > value + delta))

                solver.add(z3.Or(block))
                stats["iterations"] += 1
            else:
                break

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.
        
        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.ENUMERATION, SamplingMethod.DIKIN_WALK}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.
        
        Returns:
            A set of supported logics
        """
        return {Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA}


class LIASampler(Sampler):

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.conjuntion_sampler = None
        self.number_samples = 0

    def sample(self, number=1):
        """
        External interface
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError
