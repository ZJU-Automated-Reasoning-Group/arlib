# coding: utf-8
""" Formula manager for the CDCL(T) SMT engine

The main class is SMTPreprocess, which is used to
  - Convert an SMT formula fml to CNF (using Z3's tactics)
  - Build the Boolean skeleton P of the SMT formula fml
  - ...

"""

# import itertools
from typing import List

import z3

from arlib.cdclt.cdclt_config import m_init_abstraction
from arlib.cdclt.cdclt_config import InitAbstractionStrategy
from arlib.utils import SolverResult


# logger = logging.getLogger(__name__)


def extract_literals_square(clauses: List) -> List[List]:
    """
    After calling  clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
    the object clauses is of the following form
        [Or(...), Or(...), Or(...), ...]
    but not the usual "CNF" form (which is our initial goal)
        [[...], [...], [...]]
    Thus, this function aims to build such a CNF
    """
    maximum_models = 1  # estimate the maximum possible models (a naive strategy)
    res = []
    for cls in clauses:
        if z3.is_or(cls):
            tmp_cls = []
            for lit in cls.children():
                tmp_cls.append(lit)
            res.append(tmp_cls)
            maximum_models *= len(tmp_cls)
        else:
            res.append([cls])
    print("Maximum possible models: ", maximum_models)
    return res


class BooleanFormulaManager(object):
    """
    Track the correlations between Boolean variables and theory atoms
    NOTE: Currently, the manager does not maintain any z3-specific objects (e.g., context, expr),
          because it is used for inter-process communication
    """

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial cnt in SMT2 (without "assert")

        self.numeric_clauses = []  # initial cnt in numerical clauses
        self.bool_vars_name = []  # p@0, p@1, ...
        self.num2vars = {}  # 0 -> p@0, ...
        self.vars2num = {}  # p@0 -> 1


class TheoryFormulaManager(object):
    """
    Maintain theory information
    NOTE: Currently, the manager does not maintain any z3-specific objects (e.g., context, expr),
          because it is used for inter-process communication
    """

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""


class SMTPreprocessor4Process(object):
    """
    This is the preprocessing phase of the CDCL(T)-based SMT solving engine.
    The key goal is to perform basic simplifications and convert the simplified formula
    to CNF (from which we can build the Boolean abstraction of the original SMT formula)

    NOTE: This class is used for inter-process communication, and it will produce two objects,
        namely a BooleanFormulaManager and a TheoryFormulaManager

    NOTE:
        Consider the function SMTPreprocessor4Process.from_smt2_string,
        After calling  clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
        the object clauses is of the following form
            [Or(...), Or(...), Or(...), ...]
        but not the usual "CNF" form (which is our initial goal)
            [[...], [...], [...]]

        If using m_init_abstraction = InitAbstractionStrategy.CLAUSE,
        the  "Boolean abstraction" actually maps each clause to a Boolean variable,
          but not each atom!!!
    """

    def __init__(self):
        self.index = 1
        self.status = SolverResult.UNKNOWN
        self.bool_clauses = None  # clauses of the initial Boolean abstraction

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        """Map a theory atom to a Boolean variable"""
        # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
        if atom in atom2bool:
            return atom2bool[atom]
        p = z3.Bool("p@%d" % self.index)
        self.index += 1
        atom2bool[atom] = p
        return p

    def abstract_lit(self, atom2bool, lit) -> z3.ExprRef:
        """Abstract a literal"""
        if z3.is_not(lit):
            return z3.Not(self.abstract_atom(atom2bool, lit.arg(0)))
        return self.abstract_atom(atom2bool, lit)

    def abstract_clause(self, atom2bool, clause):
        return z3.Or([self.abstract_lit(atom2bool, lit) for lit in clause])

    def abstract_clauses(self, atom2bool, clauses):
        return [self.abstract_clause(atom2bool, clause) for clause in clauses]

    def build_numeric_clauses(self, bool_manager):
        """Currently, self.bool_clauses uses string
        But in some cases, we need to obtain numeric clauses
        """
        # assert m_init_abstraction == InitAbstractionStrategy.ATOM
        # print(self.bool_clauses)
        for cls in self.bool_clauses:
            if z3.is_or(cls):
                tmp_cls = []
                for lit in cls.children():
                    if z3.is_not(lit):
                        tmp_cls.append(-bool_manager.vars2num[str(lit.children()[0])])
                    else:
                        tmp_cls.append(bool_manager.vars2num[str(lit)])
                bool_manager.numeric_clauses.append(tmp_cls)
            else:
                # unary clause
                if z3.is_not(cls):
                    cls.append(-bool_manager.vars2num[str(cls.children()[0])])
                else:
                    cls.append(bool_manager.vars2num[str(cls)])

    def from_smt2_string(self, smt2string: str):
        # fml = z3.And(z3.parse_smt2_file(filename))
        fml = z3.And(z3.parse_smt2_string(smt2string))
        # clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
        clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
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

        # the name abs is not good
        g_atom2bool = {}
        self.bool_clauses = self.abstract_clauses(g_atom2bool, clauses)

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(after_simp)
        s.add(self.bool_clauses)

        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            if "p@" in line:
                bool_manager.smt2_signature.append(line)
            th_manager.smt2_signature.append(line)

        # initialize some mappings
        bool_var_id = 1
        for atom in g_atom2bool:
            bool_var = str(g_atom2bool[atom])
            bool_manager.num2vars[bool_var_id] = bool_var
            bool_manager.vars2num[bool_var] = bool_var_id
            bool_manager.bool_vars_name.append(bool_var)
            bool_var_id += 1

        # initialize some cnt
        bool_manager.smt2_init_cnt = z3.And(self.bool_clauses).sexpr()
        theory_init_fml = z3.And([p == g_atom2bool[p] for p in g_atom2bool])
        th_manager.smt2_init_cnt = theory_init_fml.sexpr()

        # NOTE: only useful when using special Boolean engines
        self.build_numeric_clauses(bool_manager)

        return bool_manager, th_manager


class SMTPreprocessor4Thread(object):
    """
    This is the preprocessing phase of the CDCL(T)-based SMT solving engine.
    The key goal is to perform basic simplifications and convert the simplified formula
    to CNF (from which we can build the Boolean abstraction of the original SMT formula)

    NOTE: This class is used for inter-thread communication

    """

    def __init__(self):
        self.index = 1
        self.main_z3_ctx = None
        self.status = SolverResult.UNKNOWN
        self.bool_abstraction = None  # clauses of the initial Boolean abstraction
        self.init_theory_fml = None
        self.bool_variables = []

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        """Map a theory atom to a Boolean variable"""
        # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
        if atom in atom2bool:
            return atom2bool[atom]
        p = z3.Bool("p@%d" % self.index)
        self.index += 1
        atom2bool[atom] = p
        return p

    def abstract_lit(self, atom2bool, lit) -> z3.ExprRef:
        """Abstract a literal"""
        if z3.is_not(lit):
            return z3.Not(self.abstract_atom(atom2bool, lit.arg(0)))
        return self.abstract_atom(atom2bool, lit)

    def abstract_clause(self, atom2bool, clause):
        return z3.Or([self.abstract_lit(atom2bool, lit) for lit in clause])

    def abstract_clauses(self, atom2bool, clauses):
        return [self.abstract_clause(atom2bool, clause) for clause in clauses]

    def from_smt2_string(self, smt2string: str):
        # fml = z3.And(z3.parse_smt2_file(filename))
        fml = z3.And(z3.parse_smt2_string(smt2string))
        self.main_z3_ctx = fml.ctx  # keep tracking the main Z3 thread
        # clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
        clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
        after_simp = clauses.as_expr()
        if z3.is_false(after_simp):
            self.status = SolverResult.UNSAT
        elif z3.is_true(after_simp):
            self.status = SolverResult.SAT

        if self.status != SolverResult.UNKNOWN:
            return None, None

        if m_init_abstraction == InitAbstractionStrategy.ATOM:
            clauses = extract_literals_square(clauses[0])

        # the name abs is not good
        g_atom2bool = {}
        self.bool_abstraction = z3.And(self.abstract_clauses(g_atom2bool, clauses))

        self.init_theory_fml = z3.And([p == g_atom2bool[p] for p in g_atom2bool])

        self.bool_variables = [g_atom2bool[p] for p in g_atom2bool]
