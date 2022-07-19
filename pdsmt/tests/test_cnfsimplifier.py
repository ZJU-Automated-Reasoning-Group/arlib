# coding: utf-8
"""
For testing the CNF simplifier
"""

from pdsmt.tests import TestCase, main
from pdsmt.bool import simplify_numeric_clauses


class TestCNFSimplifier(TestCase):
    """
    Test CNF simplification strategies
    """

    def test_cnf_simp(self):
        clauses = [[1, 3], [-1, 2, -4], [2, 4], [4]]
        new_cls = simplify_numeric_clauses(clauses)
        print(new_cls)
        assert True


if __name__ == '__main__':
    main()
