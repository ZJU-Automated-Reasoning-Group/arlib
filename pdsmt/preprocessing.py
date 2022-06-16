# coding: utf-8
"""
    Consider the function SMTPreprocess.from_smt2_string,
    After calling  clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
    the object clauses is of the following form
        [Or(...), Or(...), Or(...), ...]
    but not the usual "CNF" form (which is our initial goal)
        [[...], [...], [...]]

    If using m_init_abstraction = InitAbstractionStrategy.CLAUSE,
    the  "Boolean abstraction" actually maps each clause to a Boolean variable,
      but not each atom!!!
"""
from typing import List
import z3
from .config import m_init_abstraction
from .formula_manager import BooleanFormulaManager, TheoryFormulaManager
from .utils import SolverResult, InitAbstractionStrategy


def extract_literals_square(clauses: List) -> List[List]:
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
    """
    Perform basic simplifications and convert to CNF
    """

    def __init__(self):
        self.index = 1
        self.status = SolverResult.UNKNOWN
        self.bool_clauses = None  # clauses of the initial Boolean abstraction

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        """Map a theory atom to a Boolean variable"""
        # FIXME: should we identify the distinguish aux. vars introduced by tseitin' transformation?
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
        """TODO: improve performance?"""
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
        clauses = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'tseitin-cnf')(fml)
        # clauses = z3.Then('simplify', 'tseitin-cnf')(fml)
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

        abs = {}
        self.bool_clauses = self.abstract_clauses(abs, clauses)

        s = z3.Solver()  # a container for collecting variable signatures
        s.add(after_simp)
        s.add(z3.And(self.bool_clauses))

        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            if "p@" in line:
                bool_manager.smt2_signature.append(line)
            th_manager.smt2_signature.append(line)

        # initialize some mappings
        bool_var_id = 1
        for atom in abs:
            bool_var = str(abs[atom])
            bool_manager.num2vars[bool_var_id] = bool_var
            bool_manager.vars2num[bool_var] = bool_var_id
            bool_manager.bool_vars_name.append(bool_var)
            bool_var_id += 1

        # initialize some cnt
        bool_manager.smt2_init_cnt = z3.And(self.bool_clauses).sexpr()
        theory_init_fml = z3.And([p == abs[p] for p in abs])
        th_manager.smt2_init_cnt = theory_init_fml.sexpr()

        # NOTE: only useful when using special Boolean engines
        self.build_numeric_clauses(bool_manager)

        return bool_manager, th_manager
