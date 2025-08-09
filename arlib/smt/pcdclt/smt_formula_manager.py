"""Formula manager for the CDCL(T) SMT engine.

Provides preprocessing and Boolean/theory managers for inter-process communication.
"""
from typing import List

import z3

from arlib.smt.pcdclt.cdclt_config import InitAbstractionStrategy
from arlib.smt.pcdclt.cdclt_config import m_init_abstraction
from arlib.utils import SolverResult

def extract_literals_square(clauses: List) -> List[List]:
    """Convert Z3 Or-expr list into CNF-like list-of-lists."""
    result = []
    for clause in clauses:
        if z3.is_or(clause):
            result.append(list(clause.children()))
        else:
            result.append([clause])
    return result


class BooleanFormulaManager(object):
    """Tracks mappings between Boolean vars and theory atoms (no Z3 objects)."""

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial cnt in SMT2 (without "assert")

        self.numeric_clauses = []  # initial cnt in numerical clauses
        self.bool_vars_name = []  # p@0, p@1, ...
        self.num2vars = {}  # 0 -> p@0, ...
        self.vars2num = {}  # p@0 -> 1


class TheoryFormulaManager(object):
    """Holds theory-side data (no Z3 objects)."""

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""


class SMTPreprocessor4Process(object):
    """Preprocess, simplify, and build Boolean abstraction and theory skeleton."""

    def __init__(self):
        self.index = 1
        self.status = SolverResult.UNKNOWN
        self.bool_clauses = None  # clauses of the initial Boolean abstraction

    def abstract_atom(self, atom2bool, atom) -> z3.ExprRef:
        """Map a theory atom to a Boolean variable"""
        # FIXME: should we identify and distinguish aux. vars introduced by tseitin' transformation?
        if atom in atom2bool:
            return atom2bool[atom]
        pred = z3.Bool("p@%d" % self.index)
        self.index += 1
        atom2bool[atom] = pred
        return pred

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
        """Convert Boolean Z3 clauses to numeric clauses for SAT solving."""
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
                tmp_cls = []
                if z3.is_not(cls):
                    tmp_cls.append(-bool_manager.vars2num[str(cls.children()[0])])
                else:
                    tmp_cls.append(bool_manager.vars2num[str(cls)])
                bool_manager.numeric_clauses.append(tmp_cls)

    def from_smt2_string(self, smt2string: str):
        fml = z3.And(z3.parse_smt2_string(smt2string))
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

        g_atom2bool = {}
        self.bool_clauses = self.abstract_clauses(g_atom2bool, clauses)

        sol = z3.Solver()  # collect variable signatures
        sol.add(after_simp)
        sol.add(self.bool_clauses)

        # FIXME: currently, the theory solver also uses the Boolean variables
        for line in sol.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            if "p@" in line:
                bool_manager.smt2_signature.append(line)
            th_manager.smt2_signature.append(line)

        bool_var_id = 1
        for atom in g_atom2bool:
            bool_var = str(g_atom2bool[atom])
            bool_manager.num2vars[bool_var_id] = bool_var
            bool_manager.vars2num[bool_var] = bool_var_id
            bool_manager.bool_vars_name.append(bool_var)
            bool_var_id += 1

        bool_manager.smt2_init_cnt = z3.And(self.bool_clauses).sexpr()
        theory_init_fml = z3.And([p == g_atom2bool[p] for p in g_atom2bool])
        th_manager.smt2_init_cnt = theory_init_fml.sexpr()

        self.build_numeric_clauses(bool_manager)

        return bool_manager, th_manager
