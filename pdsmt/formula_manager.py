# coding: utf-8
import z3
from .smtlib_solver import SMTLIBSolver
from .config import m_smt_solver_bin


# from config import SolverResult


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
        self.boolean_sig = ""
        self.theory_sig = ""

        self.bool2atom = {}  # p0 -> a + -1*b == 0
        self.num2bool = {}  # 1 -> p0

    def get_atom(self, bool: str):
        """
        TODO: better to use numeric number as the "common language" between th and bool solvers
        because we need to interact with the DIMACS world

        NOTE: tseitin-cnf may introduce additional Boolean vars, which do not correspond to
        theory atoms?
        """
        return self.bool2atom[bool]

    def debug(self):
        print(self.bool2atom)
        # print(self.boolean_sig)
        # print(self.theory_sig)
        print(self.num2bool)


class FormulaManagerBuilder(object):
    """
    FormulaManager does not have any dependence on Z3.
    But to build a manager object, we may use Z3 (as in this class)
    """

    def __init__(self):
        self.index = 0

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

    def from_smt2_file(self, filename: str) -> FormulaManager:
        fml = z3.And(z3.parse_smt2_file(filename))
        clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
        abs = {}
        boolean_abs = z3.And(self.abstract_clauses(abs, clauses))

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(fml)
        s.add(boolean_abs)
        boolean_sigs = []
        theory_sigs = []
        # FIXME: currently, the theory solver also uses the Boolean variables
        # FIXME: This might not be a good idea (e.g., it increases the size of the formula)
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            elif "p@" in line:
                boolean_sigs.append(line)
            theory_sigs.append(line)

        fml_manager = FormulaManager()

        abstract_boolean_id = 0
        for atom in abs:
            bool_name = str(abs[atom])
            fml_manager.bool2atom[bool_name] = atom.sexpr()
            # TODO: the num_id might not be good
            fml_manager.num2bool[abstract_boolean_id] = bool_name
            abstract_boolean_id += 1

        fml_manager.boolean_sig = "\n".join(boolean_sigs)
        fml_manager.theory_sig = "\n".join(theory_sigs)

        # print(abs)
        # print(clauses)
        # fml_manager.debug()

        return fml_manager, boolean_abs.sexpr()


class BooleanSolver():

    def __init__(self, fml_manager):
        self.fml_manager = fml_manager
        self.base_cnts = "(set-logic QF_FD)\n"
        self.base_cnts += self.fml_manager.boolean_sig
        # self.base_cnts

    def check_sat(self):
        smt2sting = "(set-logic QF_BV)\n"
        bin_solver = SMTLIBSolver(m_smt_solver_bin)
        res = bin_solver.check_sat_from_scratch(smt2sting)
        bin_solver.stop()
        print(res)


class TheorySolver():

    def __init__(self, fml_manager):
        self.fml_manager = fml_manager
        self.base_cnts = "(set-logic QF_BV)\n"
        self.base_cnts += self.fml_manager.boolean_sig

    def check_sat_assuming(self, lits):
        smt2sting = ""

        for lit in lits:
            smt2sting += "(assert {})\n".format(self.fml_manager.get_atom(lit))

        print(smt2sting)


def simple_cdclt(filename):
    # prop_solver.add(abstract_clauses(abs, clauses))
    # theory_solver.add([p == abs[p] for p in abs])
    fml_manager, boolean_abs = FormulaManagerBuilder().from_smt2_file(filename)
    print(boolean_abs)
    # TODO: need to build the initial constraints

    bool_solver = BooleanSolver(fml_manager)
    theory_solver = TheorySolver(fml_manager)

    """
    prop_solver = z3.SolverFor("QF_FD")
    theory_solver = z3.SolverFor("QF_BV")
    abs = {}
    prop_solver.add(abstract_clauses(abs, clauses))
    theory_solver.add([p == abs[p] for p in abs])
    while True:
        is_sat = prop_solver.check()
        if z3.sat == is_sat:
            m = prop_solver.model()
            lits = [mk_lit(m, abs[p]) for p in abs]
            if z3.unsat == theory_solver.check(lits):
                # FIXME: use the naive "blocking formula" or use unsat core to refine
                # If unsat_core is enabled, the bit-vector solver might be slow
                # prop_solver.add(Not(And(theory_solver.unsat_core())))
                prop_solver.add(z3.Not(z3.And(lits)))
                # print(prop_solver)
            else:
                # print(theory_solver.model())
                print("sat")
                return
        else:
            # print(is_sat)
            print("unsat")
            return
    """
