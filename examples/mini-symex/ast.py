from z3 import *
import abc
import functools
import operator

class ast_base(object):
  __metaclass__ = abc.ABCMeta

  def __str__(self):
    return str(self._z3expr())

  @abc.abstractmethod
  def __eq__(self, o):
    pass

  @abc.abstractmethod
  def __hash__(self):
    pass

  @abc.abstractmethod
  def _z3expr(self):
    ## codegen of z3 expression.
    pass

def z3expr(o):
  assert isinstance(o, ast_base)
  return o._z3expr()

class ast_func_apply(ast_base):
  def __init__(self, *args):
    for a in args:
      if not isinstance(a, ast_base):
        raise Exception("Passing a non-AST node %s %s as argument to %s" % \
                        (a, type(a), type(self)))
    self.args = args

  def __eq__(self, o):
    if type(self) != type(o):
      return False
    if len(self.args) != len(o.args):
      return False
    return all(sa == oa for (sa, oa) in zip(self.args, o.args))

  def __hash__(self):
    return functools.reduce(operator.xor, [hash(a) for a in self.args], 0)

class ast_unop(ast_func_apply):
  def __init__(self, a):
    super(ast_unop, self).__init__(a)

  @property
  def a(self):
    return self.args[0]

class ast_binop(ast_func_apply):
  def __init__(self, a, b):
    super(ast_binop, self).__init__(a, b)

  @property
  def a(self):
    return self.args[0]

  @property
  def b(self):
    return self.args[1]

class ast_const_int(ast_base):
  def __init__(self, i):
    self.i = i

  def __eq__(self, o):
    if not isinstance(o, ast_const_int):
      return False
    return self.i == o.i

  def __hash__(self):
    return hash(self.i)

  def _z3expr(self):
    return self.i

class ast_const_bool(ast_base):
  def __init__(self, b):
    self.b = b

  def __eq__(self, o):
    if not isinstance(o, ast_const_bool):
      return False
    return self.b == o.b

  def __hash__(self):
    return hash(self.b)

  def _z3expr(self):
    return self.b

class ast_eq(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) == z3expr(self.b)

class ast_and(ast_func_apply):  # logical and
  def _z3expr(self):
    return z3.And(*[z3expr(a) for a in self.args])

class ast_or(ast_func_apply):  # logical or
  def _z3expr(self):
    return z3.Or(*[z3expr(a) for a in self.args])

class ast_not(ast_unop):
  def _z3expr(self):
    return z3.Not(z3expr(self.a))

class ast_int(ast_base):
  def __init__(self, id):
    self.id = id

  def __eq__(self, o):
    if not isinstance(o, ast_int):
      return False
    return self.id == o.id

  def __hash__(self):
    return hash(self.id)

  def _z3expr(self):
    ## XXX: instead of `z3.Int(self.id)`, we model integer
    ## as a bit vector with 32 bits.
    return z3.BitVec(self.id, 32)

class ast_lt(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) < z3expr(self.b)

class ast_gt(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) > z3expr(self.b)

class ast_plus(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) + z3expr(self.b)

class ast_minus(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) - z3expr(self.b)

class ast_mul(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) * z3expr(self.b)

class ast_div(ast_binop):  # floordiv
  def _z3expr(self):
    return z3expr(self.a) / z3expr(self.b)

class ast_mod(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) % z3expr(self.b)

class ast_lshift(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) << z3expr(self.b)

class ast_rshift(ast_binop):
  def _z3expr(self):
    return z3expr(self.a) >> z3expr(self.b)

class ast_bwand(ast_binop):  # bitwise and
  def _z3expr(self):
    return z3expr(self.a) & z3expr(self.b)

class ast_bwor(ast_binop):  # bitwise or
  def _z3expr(self):
    return z3expr(self.a) | z3expr(self.b)