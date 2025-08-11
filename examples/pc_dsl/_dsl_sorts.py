from __future__ import annotations

from typing import Any

import z3

__all__ = [
    "resolve_sort",
    "BV",
    "FP",
    "Array",
    "U",
    "BVVal",
    "FPVal",
    "DEFAULT_RM",
]


def resolve_sort(spec: Any) -> z3.SortRef:
    if isinstance(spec, z3.SortRef):
        return spec
    if spec is bool:
        return z3.BoolSort()
    if spec is int:
        return z3.IntSort()
    if spec is float:
        return z3.RealSort()
    if spec is str:
        return z3.StringSort()
    if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
        tag = spec[0].lower()
        if tag == "bv" and len(spec) == 2:
            return z3.BitVecSort(int(spec[1]))
        if tag == "fp" and len(spec) == 3:
            return z3.FPSort(int(spec[1]), int(spec[2]))
        if tag == "array" and len(spec) == 3:
            return z3.ArraySort(resolve_sort(spec[1]), resolve_sort(spec[2]))
    raise TypeError(f"Unsupported sort specification: {spec!r}")


def BV(width: int) -> z3.BitVecSortRef:
    return z3.BitVecSort(int(width))


def FP(ebits: int, sbits: int) -> z3.FPSortRef:
    return z3.FPSort(int(ebits), int(sbits))


def Array(domain: Any, range: Any) -> z3.ArraySortRef:
    return z3.ArraySort(resolve_sort(domain), resolve_sort(range))


def U(name: str, arity: int = 0) -> z3.SortRef:
    return z3.DeclareSort(name)


def BVVal(value: int, width: int) -> z3.BitVecNumRef:
    return z3.BitVecVal(int(value), int(width))


def FPVal(value: float, ebits: int, sbits: int) -> z3.FPNumRef:
    return z3.FPVal(value, z3.FPSort(int(ebits), int(sbits)))


DEFAULT_RM = z3.RNE()
