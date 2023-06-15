# coding: utf-8
"""
This is a simplified, sequential version of the CDCL(T)-based SMT solving engine.

It may serve as a reference implementation of the main enigne.
"""
import logging
import re
import sys
from typing import List

from arlib.cdclt import BooleanFormulaManager, SMTPreprocessor4Process
from arlib.config import m_smt_solver_bin
from arlib.cdclt.theory import SMTLibTheorySolver
from arlib.utils import SolverResult, RE_GET_EXPR_VALUE_ALL
from arlib.utils.smtlib_solver import SMTLIBSolver

logger = logging.getLogger(__name__)


class SMTLibBoolSolver:
    """
    This implementation is brittle. Particularly, it uses an SMT solver to
    solve Boolean formulas, making the interaction both convenient (in terms of unsat cores)
    and a bit inconvenient (the Boolean engine needs to track more info?)

    TODO: I think it might also be useful to call a SAT solver via "smtlib_solver"?
    """

    def __init__(self, manager: BooleanFormulaManager):
        self.fml_manager = manager
        self.bin_solver = None
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def __del__(self):
        self.bin_solver.stop()

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Boolean solver working...")
        return self.bin_solver.check_sat()

    def get_cube_from_model(self):
        """get a model and build a cube from it."""
        raw_model = self.bin_solver.get_expr_values(self.fml_manager.bool_vars_name)
        tuples_model = re.findall(RE_GET_EXPR_VALUE_ALL, raw_model)
        # e.g., [('p@0', 'true'), ('p@1', 'false')]
        return [pair[0] if pair[1].startswith("t") else "(not {})".format(pair[0]) for pair in tuples_model]


def boolean_abstraction(smt2string: str) -> List:
    """
    Only perform Boolean abstraction (e.g., for profiling)
    """
    preprocessor = SMTPreprocessor4Process()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)
    if preprocessor.status != SolverResult.UNKNOWN:
        return []
    return bool_manager.numeric_clauses



def simple_cdclt(smt2string: str):
    preprocessor = SMTPreprocessor4Process()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    bool_solver = SMTLibBoolSolver(bool_manager)
    init_bool_fml = " (set-logic QF_FD)" + " ".join(bool_manager.smt2_signature) \
                    + "(assert {})".format(bool_manager.smt2_init_cnt)
    bool_solver.add(init_bool_fml)

    theory_solver = SMTLibTheorySolver()
    # theory_solver = PySMTTheorySolver()

    # " (set-logic ALL) " +  ....
    init_theory_fml = " (set-option :produce-unsat-cores true) " +  \
                      " ".join(th_manager.smt2_signature) + \
                      "(assert {})".format(th_manager.smt2_init_cnt)

    theory_solver.add(init_theory_fml)
    # print(theory_solver.check_sat())

    logger.debug("Finish initializing bool and theory solvers")

    while True:
        try:
            is_sat = bool_solver.check_sat()
            if SolverResult.SAT == is_sat:
                assumptions = bool_solver.get_cube_from_model()
                # print(assumptions)
                if SolverResult.UNSAT == theory_solver.check_sat_assuming(assumptions):
                    # E.g., (p @ 1(not p @ 2)(not p @ 9))
                    core = theory_solver.get_unsat_core()[1:-1]
                    blocking_clauses_core = "(assert (not (and {} )))\n".format(core)
                    # the following line uses the naive "blocking formula"
                    # blocking_clauses_assumptions = "(assert (not (and " + " ".join(assumptions) + ")))\n"
                    # print(blocking_clauses_assumptions)
                    # FIXME: the following line restricts the type of the bool_solver
                    bool_solver.add(blocking_clauses_core)
                else:
                    # print("SAT (theory solver success)!")
                    return SolverResult.SAT
            else:
                # print("UNSAT (boolean solver success)!")
                return SolverResult.UNSAT
        except Exception as ex:
            print(ex)
            print(smt2string)
            # print("\n".join(theory_solver.assertions))
            sys.exit(0)
