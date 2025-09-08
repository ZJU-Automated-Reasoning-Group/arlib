from __future__ import annotations

import z3

from arlib.symabs.congruence.congruence_abstraction import congruent_closure


def parity_example(width: int = 4) -> None:
    # Simple parity program fragment: x' is x shifted; p' toggles.
    # We only model a single step to keep it tiny.
    xs = [z3.Bool(f"x{i}") for i in range(width)]
    ps = [z3.Bool(f"p{i}") for i in range(width)]
    # Toy relation: p1 = x0 XOR p0, p2 = x1 XOR p1, ... (ripple parity)
    cnstrs = []
    for i in range(1, width):
        cnstrs.append(ps[i] == z3.Xor(xs[i - 1], ps[i - 1]))
    phi = z3.And(*cnstrs) if cnstrs else z3.BoolVal(True)

    sys = congruent_closure(phi, xs + ps, modulus=1 << 1)  # modulo 2
    print("Derived system (mod 2):", sys)


if __name__ == "__main__":
    parity_example()
