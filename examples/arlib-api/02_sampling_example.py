#!/usr/bin/env python3
"""
Model Sampling Example - Advanced Sampling Techniques

This example demonstrates arlib's model sampling capabilities across various
logics and sampling methods. Unlike simple model enumeration, sampling provides
sophisticated techniques for exploring the solution space efficiently.

arlib provides much richer sampling functionalities compared to basic Z3 usage,
including MCMC, region-based sampling, and hash-based sampling.
"""

import z3
from arlib.sampling import (
    sample_models_from_formula, 
    Logic, 
    SamplingOptions, 
    SamplingMethod,
    create_sampler
)


def boolean_sampling_example():
    """Demonstrate sampling from Boolean formulas."""
    print("=== Boolean Sampling ===")
    
    a, b, c, d = z3.Bools('a b c d')
    formula = z3.And(z3.Or(a, b, c), z3.Or(z3.Not(a), d), 
                     z3.Or(z3.Not(b), z3.Not(c)), z3.Or(a, z3.Not(d)))
    
    for method in [SamplingMethod.ENUMERATION, SamplingMethod.HASH_BASED]:
        try:
            options = SamplingOptions(method=method, num_samples=3, random_seed=42)
            result = sample_models_from_formula(formula, Logic.QF_BOOL, options)
            print(f"{method.value}: {len(result)} samples")
        except Exception as e:
            print(f"{method.value}: not available")


def linear_arithmetic_sampling():
    """Demonstrate sampling from linear arithmetic formulas."""
    print("\n=== Linear Real Arithmetic ===")
    
    x, y, z = z3.Reals('x y z')
    formula = z3.And(x + y + z <= 10, x >= 0, y >= 0, z >= 0, 
                     2*x + y <= 8, x + 3*y <= 12)
    
    for method in [SamplingMethod.ENUMERATION, SamplingMethod.REGION, SamplingMethod.DIKIN_WALK]:
        try:
            options = SamplingOptions(method=method, num_samples=2, random_seed=42)
            result = sample_models_from_formula(formula, Logic.QF_LRA, options)
            print(f"{method.value}: {len(result)} samples")
        except Exception as e:
            print(f"{method.value}: not available")


def integer_arithmetic_sampling():
    """Demonstrate sampling from integer arithmetic formulas."""
    print("\n=== Linear Integer Arithmetic ===")
    
    x, y = z3.Ints('x y')
    formula = z3.And(x + 2*y <= 10, x >= 0, y >= 0, x <= 5, y <= 4, x + y >= 2)
    
    options = SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=5, random_seed=42)
    result = sample_models_from_formula(formula, Logic.QF_LIA, options)
    print(f"Generated {len(result)} samples")


def mcmc_sampling_example():
    """Demonstrate MCMC sampling for complex formulas."""
    print("\n=== MCMC Sampling ===")
    
    x, y, z = z3.Reals('x y z')
    formula = z3.And(x*x + y*y <= 4, z >= x + y, z <= 3, x >= -2, y >= -2, z >= -2)
    
    try:
        options = SamplingOptions(method=SamplingMethod.MCMC, num_samples=3, 
                                random_seed=42, burn_in=100, thin=10)
        result = sample_models_from_formula(formula, Logic.QF_NRA, options)
        print(f"MCMC generated {len(result)} samples")
    except Exception as e:
        print("MCMC sampling not available")


def custom_sampler_example():
    """Demonstrate creating and using custom samplers."""
    print("\n=== Custom Sampler ===")
    
    p, q, r = z3.Bools('p q r')
    formula = z3.And(z3.Or(p, q), z3.Or(z3.Not(p), r), z3.Or(q, z3.Not(r)))
    
    try:
        sampler = create_sampler(Logic.QF_BOOL, SamplingMethod.ENUMERATION)
        sampler.init_from_formula(formula)
        options = SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=5, random_seed=123)
        result = sampler.sample(options)
        print(f"Custom sampler generated {len(result)} samples")
    except Exception as e:
        print("Custom sampler failed")


def main():
    """Run all sampling examples."""
    print("Model Sampling Examples - arlib's Advanced Sampling")
    print("=" * 50)
    
    boolean_sampling_example()
    linear_arithmetic_sampling()
    integer_arithmetic_sampling()
    mcmc_sampling_example()
    custom_sampler_example()
    
    print("\nSampling examples completed!")


if __name__ == "__main__":
    main() 