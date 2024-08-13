from z3 import *
from typing import List

def optimize_as_long(fml: z3.ExprRef, obj: z3.ExprRef, minimize=False, timeout: int = 0):
    """
    The optimize function takes in a formula, an objective function, and whether the
     objective is to be minimized. It then adds the formula to a z3 solver object and sets
    its timeout if one was specified. It then maximizes/minimizes
    the given objective function depending on what was specified.

    :param fml:z3.ExprRef: Pass the formula
    :param obj:z3.ExprRef: Specify the objective function
    :param minimize: Specify whether we want to maximize or minimize the objective function
    :param timeout:int: Set a timeout for the optimization
    :return: The value of the objective
    """
    s = z3.Optimize()
    s.add(fml)
    if timeout > 0:
        s.set("timeout", timeout)
    if minimize:
        obj = s.minimize(obj)
    else:
        obj = s.maximize(obj)
    if s.check() == z3.sat:
        return obj.value().as_long()


def box_optimize_as_long(fml: z3.ExprRef, minimize: List, maximize: List, timeout: int = 0):
    """
    Returns a model based on given constraints as a tuple
    :param fml: formula
    :param minimize: list of minimization conditions
    :param maximize: list of maximization conditions
    :param timeout: timeout
    :return:
    """
    s = z3.Optimize()
    s.set("opt.priority", "box")
    s.add(fml)
    if timeout > 0:
        s.set("timeout", timeout)
    min_objectives = [s.minimize(e) for e in minimize]
    max_objectives = [s.maximize(e) for e in maximize]
    if s.check() == z3.sat:
        min_res = [obj.value().as_long() for obj in min_objectives]
        max_res = [obj.value().as_long() for obj in max_objectives]
        return min_res, max_res