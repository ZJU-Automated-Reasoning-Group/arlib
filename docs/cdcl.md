## Back to the Future: Revisiting DPLL(T) for Parallel SMT Solving

To reduce the dependence (and some unexpected bottleneck) of Z3. We should try not to use Z3's API.

### Related Work

https://smt-comp.github.io/2022/results/results-cloud

- https://github.com/usi-verification-and-security/SMTS/tree/cube-and-conquer-fixed
- https://github.com/usi-verification-and-security/SMTS/tree/portfolio
- cvc5-cloud https://github.com/amaleewilson/aws-satcomp-solver-sample/tree/cvc5
- Vampire portfolio: https://smt-comp.github.io/2022/system-descriptions/Vampire.pdf

### Benchmarks

https://smtlib.cs.uiowa.edu/benchmarks.shtml


### How to Install and Use

Build and install all binary solvers in the dir `bin_solvers`?

Install the required packages according to `requirements.txt`

## Some TODOs

### On Sampling of Boolean Models

Try Unigen2 and other tools for parallel and uniform sampling of Boolean models

### On Computing Small Boolean Models

Currently, we use a "dual solver based approach" (See `arlib.bool.pysat_solver, `PySATSolver:reduce_models`).
There are other approaches for computing small (even minimal) models of a SAT formula.
E.g., 
- https://github.com/francisol/py_minimal_model

### On Parallel Checking of Boolean Models

When there is only one Boolean model to be checked, we may use "portfolio approach"
to check it in parallel
