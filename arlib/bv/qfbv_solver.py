# coding: utf-8
"""
Flattening-based QF_BV solver
"""
import logging
import sys
import time

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.bv import translate_smt2formula_to_cnf
from arlib.utils import SolverResult

logger = logging.getLogger(__name__)

"""
    cadical103  = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    cadical153  = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    gluecard3   = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    gluecard4   = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    glucose3    = ('g3', 'g30', 'glucose3', 'glucose30')
    glucose4    = ('g4', 'g41', 'glucose4', 'glucose41')
    lingeling   = ('lgl', 'lingeling')
    maplechrono = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    maplecm     = ('mcm', 'maplecm')
    maplesat    = ('mpl', 'maple', 'maplesat')
    mergesat3   = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    minicard    = ('mc', 'mcard', 'minicard')
    minisat22   = ('m22', 'msat22', 'minisat22')
    minisatgh   = ('mgh', 'msat-gh', 'minisat-gh')
"""
sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
               'g4', 'lgl', 'mcb', 'mpl', 'mg3',
               'mc', 'm22', 'msh']


class QFBVSolver:
    """
    Solving QF_BV formulas by combing Z3 and pySAT
      - Z3: Translate a QF_BV formula to a SAT formula
      - pySAT: solve the translated SAT formula
    """

    def __init__(self):
        self.fml = None    # z3.ExpeRef (not used for now!)
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat
        # self.vars = []
        self.verbose = 0
        self.signed = False
        self.model = []

    def solve_smt_file(self, filepath: str):
        fml_vec = z3.parse_smt2_file(filepath)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_string(self, smt_str: str):
        fml_vec = z3.parse_smt2_string(smt_str)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_formula(self, fml: z3.ExprRef):
        return self.check_sat(fml)

    def check_sat(self, fml: z3.ExprRef):
        """
        TODO: add an option that uses Yices2 to perform bit-blasting
        :return:
        """
        qfbv_preamble = z3.AndThen(z3.With('simplify', flat_and_or=False),
                                   z3.With('propagate-values', flat_and_or=False),
                                   z3.Tactic('elim-uncnstr'),
                                   z3.With('solve-eqs', solve_eqs_max_occs=2),
                                   z3.Tactic('reduce-bv-size'),
                                   z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                                           local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
                                   # Z3 can solve a couple of extra benchmarks by using hoist_mul but the timeout in SMT-COMP is too small.
                                   # Moreover, it impacted negatively some easy benchmarks. We should decide later, if we keep it or not.
                                   # With('simplify', hoist_mul=False, som=False, flat_and_or=False),
                                   z3.Tactic('max-bv-sharing'),
                                   z3.Tactic('ackermannize_bv'),
                                   z3.Tactic('bit-blast'),
                                   # z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
                                   # With('solve-eqs', local_ctx=True, flat=False, flat_and_or=False),
                                   z3.Tactic('tseitin-cnf'),
                                   # z3.Tactic('sat')
                                   )
        qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)

        after_simp = qfbv_tactic(fml).as_expr()
        # print(after_simp)
        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        elif z3.is_true(after_simp):
            return SolverResult.SAT
        # print(".....")
        g = z3.Goal()
        g.add(after_simp)
        pos = CNF(from_string=g.dimacs())
        # pos.to_fp(sys.stdout)
        aux = Solver(name="minisat22", bootstrap_with=pos)
        # print("solving via pysat")
        if aux.solve():
            return SolverResult.SAT
        return SolverResult.UNSAT


    def bit_blast(self):
        """
        The bit_blast function converts a bit-vector formula to Boolean logic.
        It sets the `bv2bool` and `bool2id` class attributes as the mapping from BV variables to boolean expressions
        and the mapping from boolean expressions to numerical IDs, respectively.

        """
        logger.debug("Start translating to CNF...")
        # NOTICE: can be slow
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.fml)
        self.bv2bool = bv2bool
        self.bool2id = id_table
        logger.debug("  from bv to bools: {}".format(self.bv2bool))
        logger.debug("  from bool to pysat id: {}".format(self.bool2id))

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        return clauses_numeric

    def check_sat_with_model(self):
        """Check satisfiability of a bit-vector formula
        In this function, we use self.bit_blast to maintain the correlation between
        the bit-vector and Boolean variables
        """
        clauses_numeric = self.bit_blast()
        # Main difficulty: how to infer signedness of each variable
        cnf = CNF(from_clauses=clauses_numeric)
        name = "minisat22"
        try:
            start_time = time.time()
            with Solver(name=name, bootstrap_with=cnf) as solver:
                if not solver.solve():
                    return SolverResult.UNSAT
                # TODO: figure out what is the order of the vars in the boolean model
                bool_model = solver.get_model()
                logger.debug("SAT solving time: {}".format(time.time() - start_time))
                self.model = bool_model
                return SolverResult.SAT
                """
                # The following code is for building the bit-vector model
                bv_model = {}
                if not self.signed: # unsigned
                    for bv_var in self.bv2bool:
                        bool_vars = self.bv2bool[bv_var]
                        start = self.bool2id[bool_vars[0]]  # start ID
                        bv_val = 0
                        for i in range(len(bool_vars)):
                            if bool_model[i + start - 1] > 0:
                                bv_val += 2 ** i
                        bv_model[bv_var] = bv_val
                else: # signed
                    # FIXME: the following seems to be wrong
                    for bv_var in self.bv2bool:
                        bool_vars = self.bv2bool[bv_var]
                        start = self.bool2id[bool_vars[0]]  # start ID
                        bv_val = 0
                        for i in range(len(bool_vars) - 1):
                            if bool_model[i + start - 1] > 0:
                                bv_val += 2 ** i
                        if bool_model[len(bool_vars) - 1 + start - 1] > 0:
                            bv_val = -bv_val
                        bv_model[bv_var] = bv_val
                # TODO: map back to bit-vector model
                self.model = bv_model
                print(bv_model)
                """
        except Exception as ex:
            print(ex)
