# Solution Sampling for Various Constraints

NOTICE: We focus on sampling solutions from SMT formulas, not enumeration, counting, volume estimation, or optimization (though they are closely-related topics).

## Overview

This module provides various sampling algorithms for different SMT theories. The goal is to generate diverse, representative samples from the solution space of SMT formulas.

## Dependencies

### Required
- Z3 SMT Solver (core dependency)
```bash
pip install z3-solver --user
```

### Optional

- numpy: Required for continuous domain sampling (Dikin walk, etc.)
- pyeda: Required for BDD-based sampling methods
- pysmt: Enables using multiple SMT solvers as backends

## Features

### Boolean Formula
- Uniform sampling via hashing-based techniques
- MCMC-based sampling
- BDD-based sampling for small formulas
- Support for both CNF and arbitrary Boolean formulas

### Bit-Vectors
- QuickSampler implementation (ICSE'18)
- SearchTree-based sampling
- Support for both fixed and variable-width bit-vectors

### Linear Integer and Real Arithmetic (LIRA)
- Dikin walk for continuous domains
- Mixed integer-real sampling
- Support for both equalities and inequalities
- Handles both sparse and dense constraints

### Non-linear Integer and Real Arithmetic
- MCMC sampling with specialized proposals
- Region-based sampling for continuous domains
- Support for polynomial constraints


## TBD

### String Constraints
- Length-aware sampling
- Regular expression support
- Basic string operations

### Floating Points
- Special handling of NaN and Infinity
- Uniform sampling within specified precision
- Support for all rounding modes

### Algebraic Datatypes
- Basic support for simple ADTs
- Size-bounded sampling
- Constructor-aware generation

## Applications

### Quantitative Program Analysis
- Estimating execution probabilities
- ...

### Testing
- t-wise coverage sampling
- Constraint-aware test generation

## Usage

```python
from arlib.sampling import sample_formula, Logic, SamplingOptions, SamplingMethod

# Create a formula
x, y = z3.Reals('x y')
formula = z3.And(x + y > 0, x - y < 1)

# Configure sampling options
options = SamplingOptions(
    method=SamplingMethod.MCMC,
    num_samples=10,
    timeout=60
)

# Sample solutions
result = sample_formula(formula, Logic.QF_LRA, options)
print(f"Found {len(result.samples)} samples")
```

## References

1. Dutra et al., "Efficient Sampling of SAT Solutions for Testing", ICSE 2018
2. Ermon et al., "Uniform Solution Sampling Using a Constraint Solver As an Oracle", UAI 2012
3. Chakraborty et al., "A Scalable Approximate Model Counter", CP 2013