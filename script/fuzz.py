# coding: utf-8
from formula_generator import FormulaGenerator
from z3 import *
from pdsmt.formula_manager import simple_cdclt


def test():
    w, x, y, z = Ints("w x y z")
    fg = FormulaGenerator([w, x, y, z])
    smt2string = fg.generate_formula_as_str()
    simple_cdclt(smt2string)


for _ in range(13):
    test()