"""
TODO: should we perform bit-blasting, and implement this part in the bit-level?
"""
from typing import List
from enum import Enum
from random import randrange

import z3


class IncrementalMode(Enum):
    PUSHPOP = 0  # use push/pop
    ASSUMPTION = 1  # use assumption literal
    NOINC = 2  # non incremental, every time, create a new solver


m_incremental_mode = IncrementalMode.PUSHPOP


# m_incremental_mode = IncrementalMode.ASSUMPTION

class ExistsSolver(object):
    def __init__(self, cared_vars: List, phi: z3.ExprRef):
        self.x_vars = cared_vars
        self.fmls = [phi]
        self.cared_bits = []
        for var in cared_vars:
            self.cared_bits = self.cared_bits + [z3.Extract(i, i, var) == 1 for i in range(var.size())]

    def get_uniform_samples_with_xor(self, num_samples: int) -> List[z3.ModelRef]:
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
                    # TODO: maybe use assumption literal (faster than push/pop)?
                if s.check(assumption) == z3.sat:
                    models.append(s.model())
                    num_success += 1
                    if num_success == num_samples:
                        break
        return models

    def get_models(self, num_samples: int) -> List[z3.ModelRef]:
        # return self.get_uniform_samples_with_xor(num_samples)
        models = []
        s = z3.SolverFor("QF_BV")
        s.add(z3.And(self.fmls))
        if s.check() == z3.sat:
            models.append(s.model())
            if num_samples > 1:
                models = models + self.get_uniform_samples_with_xor(num_samples - 1)
        # print(models)
        return models
