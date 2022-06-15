# coding: utf-8
from . import TestCase, main


class TestConsequence(TestCase):

    def test_cons(self):
        import z3
        a, b, c, d = z3.Bools('a b c d')
        s = z3.Solver()
        s.add(z3.Implies(a, b), z3.Implies(c, d))  # background formula
        print(s.consequences([a, c],  # assumptions
                             [a, b, c, d]))  # what is implied?

        assert (1 < 2)


if __name__ == '__main__':
    main()
