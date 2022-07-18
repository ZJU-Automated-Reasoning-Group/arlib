# coding: utf-8
"""
For testing the unsat core simplifiers

NOTE: we may also use third-party CNF simplifiers to deal with the unsat cores.
"""
from pdsmt.cdcl.formula_manager import merge_unsat_cores
from pdsmt.tests import TestCase, main


class TestUnsatCore(TestCase):

    def test_core_merge(self):
        import random
        cores = []
        for _ in range(100):
            core_len = random.randint(2, 8)
            cores.append([random.randint(-10, 10) for _ in range(core_len)])
        # print(cores)
        new_cores = merge_unsat_cores(cores)
        assert len(new_cores) <= len(cores)


if __name__ == '__main__':
    main()
