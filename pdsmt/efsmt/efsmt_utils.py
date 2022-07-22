"""Some basic functions for efsmt
"""
import time
import z3

from pdsmt.global_params import z3_exec, cvc5_exec
from pdsmt.utils.smtlib_solver import SMTLIBSolver

sat_solvers = ['cadical', 'gluecard30', 'gluecard41', 'glucose30', 'glucose41', 'lingeling',
               'maplechrono', 'maplecm', 'maplesat', 'minicard', 'mergesat3', 'minisat22', 'minisat-gh']


def solve_with_bin_smt(y, phi: z3.ExprRef, logic: str, solver_name: str):
    """Call bin SMT solvers to solve exists forall"""
    smt2string = "(set-logic {})\n".format(logic)
    sol = z3.Solver()
    sol.add(z3.ForAll(y, phi))
    smt2string += sol.to_smt2()

    # TODO: build/download bin solvers in the project
    # bin_cmd = ""
    if solver_name == "z3":
        bin_cmd = z3_exec
    elif solver_name == "cvc5":
        bin_cmd = cvc5_exec + " -q --produce-models"
    else:
        bin_cmd = z3_exec

    bin_solver = SMTLIBSolver(bin_cmd)
    start = time.time()
    res = bin_solver.check_sat_from_scratch(smt2string)
    if res == "sat":
        # print(bin_solver.get_expr_values(["p1", "p0", "p2"]))
        print("External solver success time: ", time.time() - start)
        # TODO: get the model to build the invariant
    elif res == "unsat":
        print("External solver fails time: ", time.time() - start)
    else:
        print("Seems timeout or error in the external solver")
        print(res)
    bin_solver.stop()
    return res
