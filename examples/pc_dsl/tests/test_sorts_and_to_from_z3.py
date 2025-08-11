import z3

from pc_dsl.easy_z3 import Solver, BV, Array, BVVal, to_z3, from_z3


class SortsCase(Solver):
    x: int
    y: float
    b: bool
    s: str

    assert x + 1 > 0
    assert ~b | (x < 10)


def test_resolve_and_values():
    assert SortsCase.check() == z3.sat
    # Accessing fields coerces to native Python values when possible
    assert isinstance(SortsCase.x, int)
    assert isinstance(SortsCase.y, float)
    assert isinstance(SortsCase.b, bool)


def test_to_and_from_z3_roundtrip():
    # Build a small DSL expression referencing variables and convert to z3
    env = SortsCase.current_env()
    dv = SortsCase.dsl_vars()
    expr = ((dv["x"] + 2) > 5) & ~dv["b"]
    z3expr = to_z3(expr, env)
    assert isinstance(z3expr, z3.AstRef)

    # Wrap a z3 expression into DSL and ensure it can be used again
    wrapped = from_z3(z3expr)
    # Convert back using same env
    z3expr2 = to_z3(wrapped, env)
    assert z3eq(z3.simplify(z3expr), z3.simplify(z3expr2))


def z3eq(a: z3.AstRef, b: z3.AstRef) -> bool:
    s = z3.Solver()
    s.add(a != b)
    return s.check() == z3.unsat


def test_bv_and_array_via_declare_var():
    # Use scoped declarations to work with non-basic sorts without class annotations
    SortsCase.push()
    try:
        bv8 = SortsCase.declare_var("bv8", ("bv", 8))
        arr = SortsCase.declare_var("arr", ("array", ("bv", 8), int))
        dv = SortsCase.dsl_vars()
        SortsCase.add((dv["bv8"] & 0xF) == BVVal(5, 8))
        SortsCase.add(dv["arr"][0] >= 0)
        assert SortsCase.check() == z3.sat
    finally:
        SortsCase.pop()
