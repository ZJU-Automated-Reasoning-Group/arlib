# Solution Sampling for Various Constraints

NOTICE: We do sampling, not enumeration, counting, volume estimation, or optimization (they are closely-related topics)

## Dependences

The minimal require software is Z3.
You can install it via

`pip install z3-solver --user`

To enable more features, you may install other packages
~~~~
numpy: for Dikin walk, etc.
pyeda: for BDD-related staff
pysmt: for interacting with other SMT solvers
~~~~

## Features

### Boolean Formula

### Bit-Vectors

### Linear Integer and Real

### Non-linear Integer and Real

### String

### Floating Points

### Algebraic Datatypes

## Applications

### Probabilistic Program Analysis


### Sampling for t-wise Coverage


See `twise` dir (currently from https://github.com/smba/pycosa-toolbox).
The implementation is not mature.