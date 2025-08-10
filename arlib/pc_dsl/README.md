# z3 made unreasonably easy

A Python DSL that makes Z3 constraint solving even easier through class-based syntax.

## Usage

```py
from easy_z3 import Solver

class MySolver(Solver):
    # Declarations
    n: int
    x: float
    b: bool
    f: {(int, int): float}

    # Constraints
    assert n > 0
    assert f(n, n + 1) == -2 * x
    assert ~b & (f(n ** 2, 0) < x) ^ (n < 5)
    assert b >> (n == 2)  # implies

# Solve and access results
print(MySolver)
print(MySolver.n, MySolver.x)

# Dump results to current scope
()=MySolver
print(n, x, b)
```

## Supported Types

- **Basic**: `int`, `float`, `bool`
- **Bit-vectors**: `('bv', width)` or `BV(width)`
- **Strings**: `str` with `+`, `Concat(s, t)`, `Length(s)`
- **Arrays**: `('array', domain, range)` or `Array(dom, rng)`
- **Uninterpreted**: `A = U('A')`, then annotate with `A`
- **Floating-point**: `FP(ebits, sbits)`
- **Quantifiers**: `ForAll({'x': int}, body)`, `Exists({'x': int}, body)`

## Examples

```py
from easy_z3 import Solver, BV, Array, U, ForAll, Store, FP, Concat

class BitVectorSolver(Solver):
    x: ('bv', 8)
    a: ('array', ('bv', 8), ('bv', 8))
    assert (x & 0xF) == 5
    assert Store(a, 0, 7)[0] == 7

class StringSolver(Solver):
    s: str
    t: str
    assert Concat(s, t) == "ab"
    assert Length(s) == 1

A = U('A')
class UninterpretedSolver(Solver):
    a: A
    f: {A: A}
    assert f(a) == a
```

## How It Works

Uses Python metaclasses and custom namespaces to intercept class body execution, converting assertions into Z3 constraints. The `>>` and `<<` operators represent logical implication (not bit-shifting).

## Why

To demonstrate that Python's metaprogramming capabilities can create surprisingly intuitive APIs for constraint solving.
