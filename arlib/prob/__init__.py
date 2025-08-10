"""
Probabilistic reasoning utilities.

Currently provides:
- Weighted Model Counting (WMC) over propositional CNF formulas with
  DNNF-based exact evaluation and SAT-based enumeration backend.
- Stubs for Weighted Model Integration (WMI) for future extension.

Public API:
- wmc_count
- WMCBackend, WMCOptions

Example:
    from pysat.formula import CNF
    from arlib.prob import wmc_count, WMCBackend, WMCOptions

    cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
    weights = {1: 0.6, -1: 0.4, 2: 0.7, -2: 0.3, 3: 0.5, -3: 0.5}
    result = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))
    print(result)
"""

from .base import WMCBackend, WMCOptions
from .wmc import wmc_count

__all__ = [
    "WMCBackend",
    "WMCOptions",
    "wmc_count",
]
