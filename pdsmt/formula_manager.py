# coding: utf-8
import z3
from .smtlib_solver import SMTLIBSolver
from .config import m_smt_solver_bin, m_init_abstraction
from .util import SolverResult, RE_GET_EXPR_VALUE_ALL, InitAbstractionStrategy
import re
import logging

logger = logging.getLogger(__name__)

"""
    Consider the function from_smt2_string,
    After calling  clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
    the object clauses is of the following form
        [Or(...), Or(...), Or(...), ...]
    but not the usual "CNF" form (which is our initial goal)
        [[...], [...], [...]]
  
    If using m_init_abstraction = InitAbstractionStrategy.CLAUSE, 
    the  "Boolean abstraction" actually maps each clause to a Boolean variable, 
      but not each atom!!!
"""


class BooleanFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial constraints in SMT2 string

        self.numeric_clauses = []  # initial constraints in numerical clauses
        self.bool_vars_name = []
        self.num2vars = {}


class TheoryFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""


def extract_literals_square(clauses):
    """
    After calling  clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
    the object clauses is of the following form
        [Or(...), Or(...), Or(...), ...]
    but not the usual "CNF" form (which is our initial goal)
        [[...], [...], [...]]
    Thus, this function aims to build such a CNF
    """
    res = []
    for cls in clauses:
        if z3.is_or(cls):
            tmp_cls = []
            for lit in cls.children():
                tmp_cls.append(lit)
            res.append(tmp_cls)
        else:
            res.append([cls])
    return res


class SMTPreprocess(object):

    def __init__(self):
        self.index = 0
        self.status = SolverResult.UNKNOWN
        self.bool_clauses = None  # clauses of the initial Boolean abstraction

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        if atom in atom2bool:
            return atom2bool[atom]
        p = z3.Bool("p@%d" % self.index)
        self.index += 1
        atom2bool[atom] = p
        return p

    def abstract_lit(self, atom2bool, lit) -> z3.ExprRef:
        if z3.is_not(lit):
            return z3.Not(self.abstract_atom(atom2bool, lit.arg(0)))
        return self.abstract_atom(atom2bool, lit)

    def abstract_clause(self, atom2bool, clause):
        return z3.Or([self.abstract_lit(atom2bool, lit) for lit in clause])

    def abstract_clauses(self, atom2bool, clauses):
        return [self.abstract_clause(atom2bool, clause) for clause in clauses]

    def from_smt2_string(self, smt2string: str):
        """
        FIXME: this is ugly
        """
        # fml = z3.And(z3.parse_smt2_file(filename))
        fml = z3.And(z3.parse_smt2_string(smt2string))
        # clauses = z3.Then('simplify', 'solve-eqs', 'tseitin-cnf')(fml)
        clauses = z3.Then('simplify', 'tseitin-cnf')(fml)

        after_simp = clauses.as_expr()
        if z3.is_false(after_simp):
            self.status = SolverResult.UNSAT
        elif z3.is_true(after_simp):
            self.status = SolverResult.SAT

        if self.status != SolverResult.UNKNOWN:
            return None, None

        bool_manager = BooleanFormulaManager()
        th_manager = TheoryFormulaManager()

        if m_init_abstraction == InitAbstractionStrategy.ATOM:
            clauses = extract_literals_square(clauses[0])
            # print(clauses)

        abs = {}
        self.bool_clauses = self.abstract_clauses(abs, clauses)

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(after_simp)
        s.add(z3.And(self.bool_clauses))

        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            elif "p@" in line:
                bool_manager.smt2_signature.append(line)
            th_manager.smt2_signature.append(line)

        bool_var_id = 0
        for atom in abs:
            bool_name = str(abs[atom])
            bool_manager.num2vars[bool_var_id] = bool_name
            bool_manager.bool_vars_name.append(bool_name)
            bool_var_id += 1

        theory_init_fml = z3.And([p == abs[p] for p in abs])

        bool_manager.smt2_init_cnt = z3.And(self.bool_clauses).sexpr()
        th_manager.smt2_init_cnt = theory_init_fml.sexpr()

        return bool_manager, th_manager


class TheorySolver(object):

    def __init__(self, manager: TheoryFormulaManager):
        self.fml_manager = manager
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def __del__(self):
        self.bin_solver.stop()

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        return self.bin_solver.check_sat()

    def check_sat_assuming(self, assumptions):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        return self.bin_solver.get_unsat_core()


class SMTLibBoolSolver():

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
        """
        get a model and build a cube from it.
        """
        raw_model = self.bin_solver.get_expr_values(self.fml_manager.bool_vars_name)
        tuples_model = re.findall(RE_GET_EXPR_VALUE_ALL, raw_model)
        # e.g., [('p@0', 'true'), ('p@1', 'false')]
        return [pair[0] if pair[1].startswith("t") else \
                    "(not {})".format(pair[0]) for pair in tuples_model]


def boolean_abstraction(smt2string: str):
    preprocessor = SMTPreprocess()
    preprocessor.from_smt2_string(smt2string)
    if preprocessor.status != SolverResult.UNKNOWN:
        return preprocessor.status
    return preprocessor.bool_clauses


def simple_cdclt(smt2string: str):
    preprocessor = SMTPreprocess()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    bool_solver = SMTLibBoolSolver(bool_manager)
    init_bool_fml = " (set-logic QF_FD)" + " ".join(bool_manager.smt2_signature) \
                    + "(assert {})".format(bool_manager.smt2_init_cnt)
    bool_solver.add(init_bool_fml)

    theory_solver = TheorySolver(th_manager)
    init_theory_fml = " (set-logic ALL) " + " (set-option :produce-unsat-cores true) " \
                      + " ".join(th_manager.smt2_signature) + "(assert {})".format(th_manager.smt2_init_cnt)

    theory_solver.add(init_theory_fml)

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
            exit(0)
