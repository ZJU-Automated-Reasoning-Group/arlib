# Optimization Modulo Theory Solving

Baselines
- Z3 (Wrapper of API, native)
- OptiMathSAT (Wrapper of naive)
- CVC5 (Wrapper of API, native)
- PySMT-based implementation
   - A few standard algorithms using PySMT APIs,
     which allow for using different SMT oracles (e.g., z3, cvc4,..)

Prototypes
- Bit-vector optimization
  - Single objective: TACAS'16 algorithm (implement using Z3 and PySAT)
  - Boxed multi-objective