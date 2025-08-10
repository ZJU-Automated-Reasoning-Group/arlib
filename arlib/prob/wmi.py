"""
Weighted Model Integration (WMI)

Stub definitions for future extensions. The initial goal is to expose a
compatible interface with WMC but over LRA/LIA formulas with weight/density
functions. Implementation is left for future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


class Density(Protocol):
    def __call__(self, assignment: dict[str, Any]) -> float:  # pragma: no cover - placeholder
        ...


@dataclass
class WMIOptions:
    method: str = "region"  # placeholder
    num_samples: int = 1000  # placeholder


def wmi_integrate(formula: Any, density: Density, options: WMIOptions | None = None) -> float:
    raise NotImplementedError("WMI is not yet implemented.")
