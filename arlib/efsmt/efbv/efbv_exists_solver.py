"""

"""
import logging
from enum import Enum
from random import randrange
from typing import List
import random

import z3

from arlib.utils.exceptions import ExitsSolverSuccess, ExitsSolverUnknown

logger = logging.getLogger(__name__)


class IncrementalMode(Enum):
    PUSHPOP = 0  # use push/pop
    ASSUMPTION = 1  # use assumption literal
    NOINC = 2  # non incremental, every time, create a new solver


m_incremental_mode = IncrementalMode.ASSUMPTION


class ExistsSolver(object):
    def __init__(self, cared_vars, phi):
        self.x_vars = cared_vars
        self.fmls = [phi]
        self.cared_bits = []
        for var in cared_vars:
            self.cared_bits = self.cared_bits + [z3.Extract(i, i, var) == 1 for i in range(var.size())]

    def get_samples_with_rand_seeds(self, num_samples: int):
        models = []
        s = z3.SolverFor("QF_BV")
        s.add(z3.And(self.fmls))
        for _ in range(num_samples):
            s.push()
            s.set("random_seed", random.randint(1, 100))
            s.check()
            models.append(s.model())
            s.pop()
        return models

    def get_samples_with_xor_sampling(self, num_samples: int):
        """
        Get num_samples models (projected to vars)
        TODO: I think this could be run in parallel?
        """
        models = []
        s = z3.SolverFor("QF_BV")
        s.add(z3.And(self.fmls))
        num_success = 0
        if m_incremental_mode == IncrementalMode.PUSHPOP:
            while True:
                s.push()
                rounds = 3  # why 3?
                for _ in range(rounds):
                    trials = 10
                    fml = z3.BoolVal(randrange(0, 2))
                    for i in range(trials):
                        fml = z3.Xor(fml, self.cared_bits[randrange(0, len(self.cared_bits))])
                    # TODO: maybe use assumption literal (faster than push/pop)?
                    s.add(fml)
                if s.check() == z3.sat:
                    models.append(s.model())
                    num_success += 1
                    if num_success == num_samples:
                        break
                s.pop()
        elif m_incremental_mode == IncrementalMode.ASSUMPTION:
            while True:
                rounds = 3  # why 3?
                assumption = z3.BoolVal(True)
                for _ in range(rounds):
                    trials = 10
                    fml = z3.BoolVal(randrange(0, 2))
                    for _ in range(trials):
                        fml = z3.Xor(fml, self.cared_bits[randrange(0, len(self.cared_bits))])
                    assumption = z3.And(assumption, fml)
                if s.check(assumption) == z3.sat:
                    models.append(s.model())
                    num_success += 1
                    if num_success == num_samples:
                        break
        return models

    def get_models(self, num_samples: int) -> List[z3.ModelRef]:
        # return self.get_uniform_samples_with_xor(num_samples)
        sol = z3.SolverFor("QF_BV")
        sol.add(z3.And(self.fmls))
        res = sol.check()
        if res == z3.sat:
            if num_samples == 1:
                return [sol.model()]
            else:
                # return self.get_samples_with_rand_seeds(num_samples)
                return self.get_samples_with_xor_sampling(num_samples)
        elif res == z3.unsat:
            return []
        else:
            print(sol)
            raise ExitsSolverUnknown()