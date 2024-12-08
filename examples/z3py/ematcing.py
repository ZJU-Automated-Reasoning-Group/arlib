# Pattern-based quantifier instantiation
#
# Use:
#    python quantifier_pattern.py
#

from z3 import *

s = Solver()

# The basic idea behind pattern-based quantifier instantiation is in a sense straight-forward:
# Annotate a quantified formula using a pattern that contains all the bound variables.
# Pattern is an expression (that does not contain binding operations, such as quantifiers)
# that contains variables bound by a quantifier.
# Then instantiate the quantifier whenever a term that matches the pattern is create during search

# Make sure that Model-based quantifier instantiation engine is disabled!
# s.set(auto_config=False, mbqi=False)

f = Function('f', IntSort(), IntSort())
g = Function('g', IntSort(), IntSort())
a, b, c = Ints('a b c')
x = Int('x')

s.set(auto_config=False, mbqi=False)
s.add(ForAll(x, f(g(x)) == x, patterns=[f(g(x))]),
      g(a) == c,
      g(b) == c,
      a != b)

# Display solver state using internal format
print("#1 Less permissive")
print(s.sexpr())
print(s.check())

print()
s.reset()

# When the more permissive pattern g(x) is used, Z3 proves the formula to be unsatisfiable.
# More restrictive patterns minimize the number of instantiations (and potentially improve performance)

s.set(auto_config=False, mbqi=False)
s.add(ForAll(x, f(g(x)) == x, patterns=[g(x)]),
      g(a) == c,
      g(b) == c,
      a != b)

# Display solver state using internal format
print("#2 More permissive")
print(s.sexpr())
print(s.check())

print()
s.reset()

# Some patterns may also create long instantiation chains.
# Consider following assertion
#
# ForAll([x, y], Implies(subtype(x, y), subtype(array_of(x), array_of(y))), patterns=[subtype(x,y)])
#
# The axiom gets instantiated whenever there is some ground term of the form subtype(s, t).
# The instantiation causes a fresh ground term subtype(array_of(s), array_of(t)), which enables a new instantiation
# This undesirable situation is called a matching loop. Z3 uses many heuristics to break matching loops.

# What defines the terms that are created during search?
# In the context of most SMT solvers, and of the Simplify theorem prover, terms exist as part of the input formula,
# they are of course also created by instantiating quantifiers, but terms are also implicitly created when
# equalities are asserted. The last point means that terms are considered up to congruence and pattern matching
# takes place modulo ground equalities.
# We call the matching problem E-matching.

s.set(auto_config=False, mbqi=False)
s.add(ForAll(x, f(g(x)) == x, patterns=[f(g(x))]),
      a == g(b),
      b == c,
      f(a) != c)

# The terms f(a) and f(g(b)) are equal modulo the equalities.
# The pattern f(g(x)) can be matched and x bound to b and the equality f(g(b)) == b is deduced.
