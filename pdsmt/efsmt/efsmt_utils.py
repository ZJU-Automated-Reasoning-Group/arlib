"""Some basic functions for efsmt
"""
import time
from typing import List

import z3
from z3.z3util import get_vars

from ..global_params import z3_exec, cvc5_exec
from ..utils.smtlib_solver import SMTLIBSolver


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


def simple_cegar_efsmt(logic: str, y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    """ Solves exists x. forall y. phi(x, y) with simple CEGAR
    """
    x = [item for item in get_vars(phi) if item not in y]
    # set_param("verbose", 15)
    # set_param("smt.arith.solver", 3)
    qf_loigc = ""
    if "IA" in logic:
        qf_logic = "QF_LIA"
    elif "RA" in logic:
        qf_loigc = "QF_LRA"
    elif "BV" in logic:
        qf_loigc = "QF_BV"

    if qf_loigc != "":
        esolver = z3.SolverFor(qf_logic)
        fsolver = z3.SolverFor(qf_loigc)
    else:
        esolver = z3.Solver()
        fsolver = z3.Solver()

    esolver.add(z3.BoolVal(True))

    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("round: ", loops)
        eres = esolver.check()
        if eres == z3.unsat:
            return z3.unsat
        else:
            emodel = esolver.model()
            mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
            sub_phi = z3.simplify(z3.substitute(phi, mappings))
            fsolver.push()
            fsolver.add(z3.Not(sub_phi))
            if fsolver.check() == z3.sat:
                fmodel = fsolver.model()
                y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                esolver.add(sub_phi)
                fsolver.pop()
            else:
                return z3.sat
    return z3.unknown
