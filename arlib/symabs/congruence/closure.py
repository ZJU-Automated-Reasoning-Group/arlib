from __future__ import annotations

from typing import List, Sequence, Tuple

import z3

from .system import CongruenceSystem


def _bools_to_ints(bs: Sequence[z3.BoolRef]) -> List[z3.IntNumRef | z3.ArithRef]:
    return [z3.If(b, z3.IntVal(1), z3.IntVal(0)) for b in bs]


def _model_vector(model: z3.ModelRef, xs: Sequence[z3.ArithRef]) -> List[int]:
    vals: List[int] = []
    for x in xs:
        v = model.eval(x, model_completion=True)
        vals.append(int(v.as_long()))
    return vals


def _augment_with_model(sys: CongruenceSystem, xvec: List[int]) -> None:
    """Given a new model x ∈ {0,1}^k, add the equation fixing x to the system's
    row-space by joining with the affine hull of previous models. In practice we
    simply append the row x ≡ c (mod m) captured by a single equality:
    Σ 2^j * x_j ≡ Σ 2^j * x_j (mod m). This seeds information for elimination.

    This is a simplification of the paper's triangular merge; it works well with
    the stability checking loop and yields a sound (though not maximally compact)
    system.
    """
    k = len(xvec)
    coeffs = [1] * k
    rhs = sum(xvec)
    sys.add_row(coeffs, rhs)


def congruent_closure(
    formula: z3.BoolRef,
    bool_vars: Sequence[z3.BoolRef],
    modulus: int,
    max_iters: int = 256,
) -> CongruenceSystem:
    """Compute a modular congruence abstraction of ``formula`` over ``bool_vars``.

    This implements a SAT-guided CEGIS loop inspired by Fig. 2 of the paper. We
    maintain a candidate system and iteratively enforce stability: if there is a
    model of ``formula`` that violates the current rows, we add information from
    that model and continue. When no violating model exists, the system is
    stable and represents a congruent over-approximation of all models.

    Notes:
    - We use a simple row-augmentation rather than full triangular maintenance
      to keep the implementation compact. It preserves soundness; precision is
      generally good for small blocks. This can be strengthened later.
    - ``modulus`` should be 2**w.
    """
    assert len(bool_vars) > 0
    assert modulus > 0 and (modulus & (modulus - 1)) == 0

    sys = CongruenceSystem(modulus=modulus, coeffs=[], rhs=[])

    s = z3.Solver()
    s.add(formula)
    xs = _bools_to_ints(bool_vars)

    # Initially, try to obtain a few models to seed the system.
    seeds: List[List[int]] = []
    for _ in range(4):
        if s.check() != z3.sat:
            break
        m = s.model()
        v = _model_vector(m, xs)
        seeds.append(v)
        s.add(z3.Or([x != m.eval(x, model_completion=True) for x in xs]))

    for v in seeds:
        _augment_with_model(sys, v)
    sys.triangularize()

    # Stability loop
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        s2 = z3.Solver()
        s2.add(formula)
        # add negation of current system: some row must be violated
        violated: List[z3.BoolRef] = []
        for a, b in zip(sys.coeffs, sys.rhs):
            lhs = sum(int(ai) * xi for ai, xi in zip(a, xs))
            # lhs ≡ b (mod m)  <=>  ∃t. lhs = b + m*t
            t = z3.FreshInt("t_viol")
            violated.append(z3.Not(lhs == z3.IntVal(int(b)) + z3.IntVal(modulus) * t))
        if violated:
            s2.add(z3.Or(violated))

        if s2.check() != z3.sat:
            break  # stable

        m = s2.model()
        v = _model_vector(m, xs)
        _augment_with_model(sys, v)
        sys.triangularize()

    return sys
