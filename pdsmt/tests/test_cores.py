# coding: utf-8
import pytest

from . import TestCase, main
from pdsmt.formula_manager import merge_unsat_cores


class TestUnsatCore(TestCase):

    def test(self):
        import random
        cores = []
        for _ in range(1000):
            core_len = random.randint(2, 8)
            cores.append([random.randint(-10, 10) for _ in range(core_len)])
        # print(cores)
        print(len(cores))
        new_cores = merge_unsat_cores(cores)
        print(len(new_cores))

        assert True


if __name__ == '__main__':
    main()