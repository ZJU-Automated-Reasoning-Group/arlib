import z3

from pc_dsl.easy_z3 import Solver, get_constraints


class Basic(Solver):
    n: int
    m: int
    b: bool

    assert n > m
    assert m >= 0
    # Use >> as logical implication
    assert b >> (n >= m)


def test_check_and_model():
    assert Basic.check() == z3.sat
    assert Basic.model() is not None

    n_val = Basic.n
    m_val = Basic.m
    # Ints are converted to Python ints
    assert isinstance(n_val, int)
    assert isinstance(m_val, int)
    assert n_val >= m_val >= 0

    # After solving, repr should not be unsolved
    assert "unsolved" not in repr(Basic)

    # Constraints are exposed
    cs = get_constraints(Basic)
    assert isinstance(cs, tuple)
    assert len(cs) == 3
    assert all(isinstance(c, z3.AstRef) for c in cs)


def test_dsl_vars_and_add():
    # Reset and add an extra constraint using DSL vars
    Basic.reset()
    assert "unsolved" in repr(Basic)
    dv = Basic.dsl_vars()
    Basic.add(dv["n"] > dv["m"])  # adds one more assertion
    assert len(Basic.current_assertions()) == 4  # original 3 + 1
    assert Basic.check() == z3.sat


if __name__ == "__main__":
    test_check_and_model()
    test_dsl_vars_and_add()
