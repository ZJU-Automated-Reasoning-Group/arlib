# coding: utf-8
from z3 import *

# Some example about bv solvers.

t1 = AndThen(With('simplify', blast_distinct=True, elim_and=False, flat=False, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('elim-uncnstr'),
             Tactic('solve-eqs'),
             Tactic('max-bv-sharing'),
             Tactic('bit-blast'),
             With('propagate-values', push_ite_bv=False),
             Tactic('smt'))

t2 = AndThen(With('simplify', blast_distinct=False, elim_and=True, flat=False, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('elim-uncnstr'),
             Tactic('purify-arith'),
             Tactic('smt'))

t3 = AndThen(With('simplify', blast_distinct=True, elim_and=False, flat=True, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('max-bv-sharing'),
             Tactic('bit-blast'),
             Tactic('sat'))

x, y = BitVecs("x y", 8)
fml = And(x + y == 6, x - y == 2)
sol = t1.solver()
sol.add(fml)
