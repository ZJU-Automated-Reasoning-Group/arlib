# coding: utf-8
import z3
from .smtlib_solver import SMTLIBSolver
from .config import m_smt_solver_bin
from .util import SolverResult, RE_GET_EXPR_VALUE_ALL
import re
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class FormulaManager(object):
    """
    Formula manger used by theory solvers and Boolean solvers
    NOTICE
    + The initial Boolean abstraction
    + The mapping between Booleans vars and theory atoms
    + The mapping Booleans and some numeric numbers (?)

    TODO:
    - Complex clause sharing among different managers
    """

    def __init__(self):
        self.boolean_sig = []
        self.theory_sig = []

        self.bool2atom = {}  # p0 -> a + -1*b == 0
        self.num2bool = {}  # 1 -> p0
        self.bool_vars = []
        # self.aux_bool_vars = [] # created by tseitin-cnf

    def debug(self):
        print("Bool to Atom: ", self.bool2atom)


class SMTPreprocess(object):
    """
    FormulaManager does not have any dependence on Z3.
    But to build a manager object, we may use Z3 (as in this class)
    """

    def __init__(self):
        self.index = 0
        self.solved = False

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

    def from_smt2_string(self, smt2string: str) -> FormulaManager:
        fml_manager = FormulaManager()  #

        # fml = z3.And(z3.parse_smt2_file(filename))
        fml = z3.And(z3.parse_smt2_string(smt2string))
        # clauses = z3.Then('simplify', 'solve-eqs', 'tseitin-cnf')(fml)
        clauses = z3.Then('simplify', 'tseitin-cnf')(fml)

        after_simp = clauses.as_expr()
        if z3.is_false(after_simp) or z3.is_true(after_simp):
            self.solved = True
            return None, None, None

        abs = {}
        boolean_abs = z3.And(self.abstract_clauses(abs, clauses))

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(after_simp)
        s.add(boolean_abs)

        # from z3.z3util import get_vars # this API could be slow (I have experience..)
        # get_vars(z3.And(s.assertions()))

        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            elif "p@" in line:
                fml_manager.boolean_sig.append(line)
            fml_manager.theory_sig.append(line)

        bool_var_id = 0
        for atom in abs:
            bool_name = str(abs[atom])
            fml_manager.bool2atom[bool_name] = atom.sexpr()
            fml_manager.num2bool[bool_var_id] = bool_name
            bool_var_id += 1
            fml_manager.bool_vars.append(bool_name)

        # theory_init_fmls = [(p == abs[p]).sexpr() for p in abs]
        theory_init_fml = z3.And([p == abs[p] for p in abs])
        return fml_manager, boolean_abs.sexpr(), theory_init_fml.sexpr()


class TheorySolver():

    def __init__(self, manager: FormulaManager):
        self.fml_manager = manager
        self.bin_solver = None
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)
        # self.debug = True
        # self.assertions = []

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.warning("Theory solver working...")
        return self.bin_solver.check_sat()

    def check_sat_assuming(self, assumptions):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.warning("Theory solver working...")
        return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        return self.bin_solver.get_unsat_core()


class BooleanSolver():

    def __init__(self, manager: FormulaManager):
        # TODO: seems fml_manger is not useful
        self.fml_manager = manager
        self.bin_solver = None
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.warning("Boolean solver working...")
        return self.bin_solver.check_sat()

    def get_cube_from_model(self):
        """
        get a model and build a cube from it.
        """
        raw_model = self.bin_solver.get_expr_values(self.fml_manager.bool_vars)
        tuples_model = re.findall(RE_GET_EXPR_VALUE_ALL, raw_model)
        # e.g., [('p@0', 'true'), ('p@1', 'false')]
        return [pair[0] if pair[1].startswith("t") else \
                    "(not {})".format(pair[0]) for pair in tuples_model]


def simple_cdclt(smt2string: str):
    # prop_solver.add(abstract_clauses(abs, clauses))
    # theory_solver.add([p == abs[p] for p in abs])
    preprocessor = SMTPreprocess()
    fml_manager, boolean_abs, theory_cnt = preprocessor.from_smt2_string(smt2string)

    logger.warning("Finish preprocessing")

    if preprocessor.solved:
        print("Solved by the preprocessor")
        return

    bool_solver = BooleanSolver(fml_manager)
    init_bool_fml = " (set-logic QF_FD)" + " ".join(fml_manager.boolean_sig) + "(assert {})".format(boolean_abs)
    bool_solver.add(init_bool_fml)

    theory_solver = TheorySolver(fml_manager)
    init_theory_fml = " (set-logic ALL) " + " (set-option :produce-unsat-cores true) " \
                      + " ".join(fml_manager.theory_sig) + "(assert {})".format(theory_cnt)

    theory_solver.add(init_theory_fml)

    logger.warning("Finish initializing bool and theory solvers")

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
                    # print(blocking_clauses_core)
                    # the following line uses the naive "blocking formula"
                    # blocking_clauses_assumptions = "(assert (not (and " + " ".join(assumptions) + ")))\n"
                    # print(blocking_clauses_assumptions)

                    bool_solver.add(blocking_clauses_core)
                else:
                    print("SAT (theory solver success)!")
                    break
            else:
                print("UNSAT (boolean solver success)!")
                break
        except Exception as ex:
            print(ex)
            print(smt2string)
            # print("\n".join(theory_solver.assertions))
            exit(0)
            break
