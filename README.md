## Back to the Future: Revisiting DPLL(T) for Parallel SMT Solving

To reduce the dependence (and some unespected bottleneck) of Z3. We should try not to use Z3's API.

Milestones

- Do not use Z3 APIs in the main engine (theory solver, boolean solver). Only use them in the builder of the formula
  manager

- Try Unigen for uniform sampling of Boolean models

- Parallel version