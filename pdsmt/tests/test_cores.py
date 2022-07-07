# coding: utf-8

from ..cdcl.formula_manager import merge_unsat_cores
from . import TestCase, main


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
