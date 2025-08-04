# Utility functions for lia_star modules
from z3 import *
from arlib.smt.lia_star import statistics

def getModel(s, X=[]):
    res = s.check()
    statistics.z3_calls += 1
    if res != sat:
        return None
    m = s.model()
    return [m.eval(x).as_long() for x in X]
