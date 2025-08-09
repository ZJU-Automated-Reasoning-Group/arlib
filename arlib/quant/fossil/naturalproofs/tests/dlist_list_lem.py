# Only importing this for writing this file as a test
import unittest

from z3 import And, Or, Not, Implies, If
from z3 import IsSubset, Union, SetIntersect, SetComplement, EmptySet

from arlib.quant.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort
from arlib.quant.fossil.naturalproofs.decl_api import Const, Consts, Function, RecFunction, AddRecDefinition, AddAxiom
from arlib.quant.fossil.naturalproofs.prover import NPSolver
import arlib.quant.fossil.naturalproofs.proveroptions as proveroptions

# Declarations
x, nil = Consts('x nil', fgsort)
nxt = Function('nxt', fgsort, fgsort)
prv = Function('prv', fgsort, fgsort)
lst = RecFunction('lst', fgsort, boolsort)
dlst = RecFunction('dlst', fgsort, boolsort)
AddRecDefinition(lst, x, If(x == nil, True, lst(nxt(x))))
AddRecDefinition(dlst, x, If(x == nil, True,
                             If(nxt(x) == nil, True,
                                And(prv(nxt(x)) == x, dlst(nxt(x))))))

# Problem parameters
goal = Implies(And(dlst(x), x == nil), lst(x))
lemma1_params = (x,)
lemma1_body = Implies(dlst(x), lst(x))
lemmas = {(lemma1_params, lemma1_body)}
# Call a natural proofs solver
npsolver = NPSolver()
# Configure the solver
npsolver.options.instantiation_mode = proveroptions.manual_instantiation
npsolver.options.terms_to_instantiate = {x, nil}
# Ask for proof
npsolution = npsolver.solve(goal, lemmas)


class DlistListLemTest(unittest.TestCase):
    def test1(self):
        self.assertFalse(npsolution.if_sat)


if __name__ == '__main__':
    unittest.main()
