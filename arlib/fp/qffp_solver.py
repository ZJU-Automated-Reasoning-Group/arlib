

import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from arlib.utils.types import SolverResult

is_bool = z3.Probe('is-propositional')

qffp_preamble = z3.AndThen(z3.With('simplify', arith_lhs=False, elim_and=True),
                           z3.Tactic('propagate-values'),
                           z3.Tactic('fpa2bv'),
                           z3.Tactic('propagate-values'),
                           z3.With('simplify', arith_lhs=False, elim_and=True),
                           z3.Tactic('ackermannize_bv'),
                           z3.Tactic('bit-blast'),
                           z3.With('simplify', arith_lhs=False, elim_and=True),
                           # FIXME: check the usage of Probe...
                           z3.If(is_bool, 'sat', 'smt')
                           # With('solve-eqs', local_ctx=True, flat=False, flat_and_or=False),
                           # z3.Tactic('tseitin-cnf')
                           )