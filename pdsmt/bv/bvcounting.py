# coding: utf-8
import z3

from ..bool.counting.satcounting import SATModelCounter
from .mapped_blast import translate_smt2formula_to_cnf_file


class BVModelCounter:

    def __init__(self):
        pass

    def count_models(self, fml: z3.ExprRef):
        """
        Bit-level counting
        """
        cnf_file = '/tmp/out.cnf'
        translate_smt2formula_to_cnf_file(fml, cnf_file)
        sat_counter = SATModelCounter()
        return sat_counter.count_models(cnf_file)

