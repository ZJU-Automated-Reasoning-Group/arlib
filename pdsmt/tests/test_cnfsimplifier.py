# coding: utf-8
from pdsmt.bool.cnfsimplifier import simplify_numeric_clauses
from pdsmt.tests import TestCase, main


class TestCNFSimplifier(TestCase):

    def test(self):
        clauses = [[1, 3], [-1, 2, -4], [2, 4], [4]]
        new_cls = simplify_numeric_clauses(clauses)
        assert True


if __name__ == '__main__':
    main()
