# coding: utf-8
from pdsmt.tests.formula_generator import FormulaGenerator
from z3 import *
from pdsmt.formula_manager import simple_cdclt, boolean_abstraction
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

def test():
    w, x, y, z = Ints("w x y z")
    fg = FormulaGenerator([w, x, y, z])
    smt2string = fg.generate_formula_as_str()
    res = simple_cdclt(smt2string)
    # res = boolean_abstraction(smt2string)
    print(res)


for _ in range(22):
    test()
print("Finished!")