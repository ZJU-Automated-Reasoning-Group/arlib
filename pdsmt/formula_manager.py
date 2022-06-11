# coding: utf-8
import z3
from .smtlib_solver import SMTLIBSolver
from .config import m_smt_solver_bin
from .util import SolverResult, convert_value, RE_GET_EXPR_VALUE_ALL
import re


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

    def get_atom(self, bool: str):
        """
        TODO:
            + better to use numeric number as the "common language" between th and bool solvers
               because we need to interact with the DIMACS world
            + tseitin-cnf may introduce additional Boolean vars, which do not correspond to
                theory atoms?
        """
        return self.bool2atom[bool]

    def debug(self):
        print(self.bool2atom)
        # print(self.boolean_sig)
        # print(self.theory_sig)
        print(self.num2bool)


class SMTPreprocess(object):
    """
    FormulaManager does not have any dependence on Z3.
    But to build a manager object, we may use Z3 (as in this class)
    """

    def __init__(self):
        self.index = 0

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        if atom in atom2bool:
            return atom2bool[atom]
        p = z3.Bool("pp%d" % self.index)
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
        fml_manager = FormulaManager()  #

        fml = z3.And(z3.parse_smt2_file(filename))
        clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
        abs = {}
        boolean_abs = z3.And(self.abstract_clauses(abs, clauses))
        theory_init_fml = z3.And([p == abs[p] for p in abs])

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(fml)
        s.add(boolean_abs)
        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            elif "pp" in line:
                fml_manager.boolean_sig.append(line)
            fml_manager.theory_sig.append(line)

        bool_var_id = 0
        for atom in abs:
            bool_name = str(abs[atom])
            fml_manager.bool2atom[bool_name] = atom.sexpr()
            fml_manager.num2bool[bool_var_id] = bool_name
            bool_var_id += 1
            fml_manager.bool_vars.append(bool_name)

        # print(abs)
        # print(clauses)
        # fml_manager.debug()
        return fml_manager, boolean_abs.sexpr(), theory_init_fml.sexpr()


class TheorySolver():

    def __init__(self, manager: FormulaManager):
        self.fml_manager = manager
        self.bin_solver = None
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        return self.bin_solver.check_sat()

    def check_sat_assuming(self, assumptions):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        return self.bin_solver.check_sat_assuming(assumptions)

    def get_unsat_core(self):
        print(self.bin_solver.get_unsat_core())


class BooleanSolver():

    def __init__(self, manager: FormulaManager):
        self.fml_manager = manager
        self.bin_solver = None
        self.bin_solver = SMTLIBSolver(m_smt_solver_bin)

    def add(self, smt2string):
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        return self.bin_solver.check_sat()

    def get_cube_from_model(self):
        """
        get a model and build a cube from it.
        """
        raw_model = self.bin_solver.get_expr_values(self.fml_manager.bool_vars)
        tuples_model = re.findall(RE_GET_EXPR_VALUE_ALL, raw_model)
        # e.g., [('pp0', 'true'), ('pp1', 'false')]
        return [pair[0] if pair[1].startswith("t") else \
                    "(not {})".format(pair[0]) for pair in tuples_model]


def simple_cdclt(filename):
    # prop_solver.add(abstract_clauses(abs, clauses))
    # theory_solver.add([p == abs[p] for p in abs])
    fml_manager, boolean_abs, theory_cnt = SMTPreprocess().from_smt2_file(filename)

    bool_solver = BooleanSolver(fml_manager)
    init_bool_fml = " (set-logic QF_FD)" + " ".join(fml_manager.boolean_sig) + "(assert {})".format(boolean_abs)
    bool_solver.add(init_bool_fml)

    theory_solver = TheorySolver(fml_manager)
    init_theory_fml = " (set-logic ALL)" + " ".join(fml_manager.theory_sig) + "(assert {})".format(theory_cnt)
    theory_solver.add(init_theory_fml)

    while True:
        is_sat = bool_solver.check_sat()
        if SolverResult.SAT == is_sat:
            assumptions = bool_solver.get_cube_from_model()
            if SolverResult.UNSAT == theory_solver.check_sat_assuming(assumptions):
                # TODO: use the naive "blocking formula" or use unsat core to refine
                #   If unsat_core is enabled, the solver might be slower
                blocking_clauses = "(assert (not (and " + " ".join(assumptions) + ")))\n"
                bool_solver.add(blocking_clauses)
            else:
                print("sat")
                break
        else:
            print("unsat")
            break


"""
    start = time.time()
    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()
        blocking_clause_queue = multiprocessing.Queue()
        prop_model_queue = multiprocessing.Queue()
        print("workers: ", multiprocessing.cpu_count())

        for _ in range(multiprocessing.cpu_count()):
            g_process_queue.append(multiprocessing.Process(target=solver_worker_api,
                                                           args=(thsolver_smt2sting,
                                                                 ppsolver_smt2sting,
                                                                 bool_vars,
                                                                 blocking_clause_queue,
                                                                 prop_model_queue,
                                                                 result_queue
                                                                 )))
        # Start all
        for p in g_process_queue:
            p.start()

        try:
            # Wait at most 300 seconds for a return
            result = result_queue.get(timeout=600)
        except multiprocessing.queues.Empty:
            result = "unknown"
        for p in g_process_queue:
            p.terminate()

        print(result)

    print(time.time() - start)
"""
