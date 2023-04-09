# Parallel SMT CDCL(T) Solving

To optimize Z3 and reduce dependencies and bottlenecks, avoid using Z3's API.

## Related Work

Refer to the following links for related work:

- [https://smt-comp.github.io/2022/results/results-cloud](https://smt-comp.github.io/2022/results/results-cloud)
- [https://github.com/usi-verification-and-security/SMTS/tree/cube-and-conquer-fixed](https://github.com/usi-verification-and-security/SMTS/tree/cube-and-conquer-fixed)
- [https://github.com/usi-verification-and-security/SMTS/tree/portfolio](https://github.com/usi-verification-and-security/SMTS/tree/portfolio)
- cvc5-cloud: [https://github.com/amaleewilson/aws-satcomp-solver-sample/tree/cvc5](https://github.com/amaleewilson/aws-satcomp-solver-sample/tree/cvc5)
- Vampire portfolio: [https://smt-comp.github.io/2022/system-descriptions/Vampire.pdf](https://smt-comp.github.io/2022/system-descriptions/Vampire.pdf)

## Benchmarks

Refer to https://smtlib.cs.uiowa.edu/benchmarks.shtml for benchmarks.

## How to Install and Use

1. Build and install all binary solvers in the `bin_solvers` directory.
2. Install the required packages according to `requirements.txt`.

## TODOs

### On Sampling of Boolean Models

Try Unigen2 and other tools for parallel and uniform sampling of Boolean models.

### On Computing Small Boolean Models

Currently, we use a dual solver-based approach (see `arlib.bool.pysat_solver`, `PySATSolver:reduce_models`). There are other approaches for computing small (even minimal) models of a SAT formula. For example, refer to https://github.com/francisol/py_minimal_model.

### On Parallel Checking of Boolean Models

When there is only one Boolean model to be checked, use a portfolio approach to check it in parallel.