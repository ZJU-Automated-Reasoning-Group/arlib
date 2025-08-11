from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import z3


@dataclass
class CongruenceSystem:
    """A·x ≡ b (mod m) over Boolean variables x ∈ {0,1}^k interpreted as Ints.

    We store rows as pairs (coeffs, rhs) where ``coeffs`` is a list of ints of
    length k and ``rhs`` is an int. ``modulus`` m is a power of two, 2**w.

    This structure is intentionally minimal. It provides Z3 encodings for
    checking and pretty-printing. Algebraic operations (join, projection, etc.)
    can be added incrementally as needed.
    """

    modulus: int
    coeffs: List[List[int]]
    rhs: List[int]

    def __post_init__(self) -> None:
        assert self.modulus > 0 and (self.modulus & (self.modulus - 1)) == 0, "m must be 2**w"
        assert len(self.coeffs) == len(self.rhs)

    @property
    def num_rows(self) -> int:
        return len(self.coeffs)

    @property
    def width(self) -> int:
        return len(self.coeffs[0]) if self.coeffs else 0

    def add_row(self, row: Sequence[int], b: int) -> None:
        if self.coeffs:
            assert len(row) == self.width
        self.coeffs.append(list(int(x) for x in row))
        self.rhs.append(int(b) % self.modulus)

    # ----------------------- Linear algebra (mod 2**w) -----------------------
    @staticmethod
    def _mod_inv_pow2(a: int, m: int) -> Optional[int]:
        """Return modular inverse of a modulo m=2**w if it exists (a odd), else None.

        Uses Newton iteration for 2-adic inverse.
        """
        if a % 2 == 0:
            return None
        # Normalize to [0, m)
        a %= m
        # Start with inverse modulo 2
        inv = 1  # since a is odd
        k = 1
        while (1 << k) < m:
            # inv' = inv * (2 - a*inv) mod 2**(2^k)
            inv = (inv * (2 - (a * inv) % (1 << k))) % (1 << k)
            k += 1
        # Lift to full modulus
        inv = (inv * (2 - (a * inv) % m)) % m
        return inv

    def triangularize(self) -> None:
        """Bring [A|b] to an approximate upper-triangular form modulo 2**w.

        We perform Gaussian elimination where possible when a pivot with odd
        coefficient exists in the current column (so modular inverse exists).
        Remaining even-columns are left as-is. Rows are reduced modulo m.
        """
        m = self.modulus
        if not self.coeffs:
            return
        nrows = len(self.coeffs)
        ncols = len(self.coeffs[0])
        # Reduce coefficients and rhs modulo m first
        for i in range(nrows):
            self.coeffs[i] = [int(c) % m for c in self.coeffs[i]]
            self.rhs[i] = int(self.rhs[i]) % m

        r = 0  # current row
        for c in range(ncols):
            # Find pivot with odd coeff in column c
            pivot = None
            for i in range(r, nrows):
                if self.coeffs[i][c] % 2 == 1:
                    pivot = i
                    break
            if pivot is None:
                continue
            # Swap pivot row up
            if pivot != r:
                self.coeffs[r], self.coeffs[pivot] = self.coeffs[pivot], self.coeffs[r]
                self.rhs[r], self.rhs[pivot] = self.rhs[pivot], self.rhs[r]
            a_rc = self.coeffs[r][c] % m
            inv = self._mod_inv_pow2(a_rc, m)
            if inv is None:
                continue
            # Normalize pivot row to have 1 at (r,c)
            if a_rc % m != 1:
                for j in range(ncols):
                    self.coeffs[r][j] = (self.coeffs[r][j] * inv) % m
                self.rhs[r] = (self.rhs[r] * inv) % m
            # Eliminate below
            for i in range(r + 1, nrows):
                factor = self.coeffs[i][c] % m
                if factor == 0:
                    continue
                for j in range(c, ncols):
                    self.coeffs[i][j] = (self.coeffs[i][j] - factor * self.coeffs[r][j]) % m
                self.rhs[i] = (self.rhs[i] - factor * self.rhs[r]) % m
            r += 1


    def as_z3(self, bool_vars: Sequence[z3.BoolRef]) -> z3.BoolRef:
        """Return a Z3 formula encoding all congruences over the given Booleans.

        Each Boolean is cast to Int via If(b,1,0) and the congruence is encoded
        as an equality over Ints modulo m by introducing an existential k: Int
        such that Σ ai*xi = b + m*k. This is sufficient for use in solvers.
        """
        assert len(bool_vars) == self.width
        xs = [z3.If(b, z3.IntVal(1), z3.IntVal(0)) for b in bool_vars]
        constraints: List[z3.BoolRef] = []
        m = self.modulus
        for idx, (a, b) in enumerate(zip(self.coeffs, self.rhs)):
            lhs = sum(int(ai) * xi for ai, xi in zip(a, xs))
            k = z3.FreshInt("k_cong")
            constraints.append(lhs == z3.IntVal(int(b)) + z3.IntVal(m) * k)
        return z3.And(constraints) if constraints else z3.BoolVal(True)

    def __str__(self) -> str:
        rows = []
        for a, b in zip(self.coeffs, self.rhs):
            rows.append(f"{a} ≡ {b} (mod {self.modulus})")
        return "[" + "; ".join(rows) + "]"
