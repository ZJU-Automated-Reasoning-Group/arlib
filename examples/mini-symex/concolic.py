from ast import *
from utils import *
from z3 import *
from collections import defaultdict
import ctypes
import functools
import queue
import sys

INT_MAX = 0x7fffffff
INT_MIN = -INT_MAX - 1

## current path constraints.
_cur_pc = None

## initial concrete values of all variables.
_vals = None

## add a path constraint.
def add_pc(e):
  global _cur_pc
  _cur_pc.append(e)

## checker flags: overflow and divided-by-zero.
_check_overflow = False
_check_divided_by_zero = False

def enable_overflow_check():
  ## enable overflow check for addition, subtraction and multiplication.
  global _check_overflow
  _check_overflow = True

def enable_divided_by_zero_check():
  ## enable divided-by-zero check for division and modulo.
  global _check_divided_by_zero
  _check_divided_by_zero = True

def checker_decorator(error):
  def decorator_impl(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
      solver.push()
      solver.add(z3expr(ast_and(*_cur_pc)))
      exp = f(*args, **kwds)
      if not exp is None:
        solver.add(exp)
        if solver.check() == sat:
          m = solver.model()
          log("\033[91m[checker] possible %s detected: %s\033[0m" % (error, m))
      solver.pop()

    return wrapper

  return decorator_impl

## https://stackoverflow.com/a/1514309
@checker_decorator("overflow")
def check_add_overflow(a, b):
  ## integer overflow detection for addition.
  overflow = ast_and(ast_gt(b, ast(0)), ast_gt(a, ast_minus(ast(INT_MAX), b)))
  underflow = ast_and(ast_lt(b, ast(0)), ast_lt(a, ast_minus(ast(INT_MIN), b)))
  return z3expr(ast_or(overflow, underflow))

@checker_decorator("overflow")
def check_minus_overflow(a, b):
  ## integer overflow detection for subtraction.
  overflow = ast_and(ast_lt(b, ast(0)), ast_gt(a, ast_plus(ast(INT_MAX), b)))
  underflow = ast_and(ast_gt(b, ast(0)), ast_lt(a, ast_plus(ast(INT_MIN), b)))
  return z3expr(ast_or(overflow, underflow))

@checker_decorator("overflow")
def check_multiply_overflow(a, b):
  ## integer overflow detection for multiplication.
  overflow = ast_gt(a, ast_div(ast(INT_MAX), b))
  underflow = ast_lt(a, ast_div(ast(INT_MIN), b))
  special1 = ast_and(ast_eq(a, ast(-1)), ast_eq(b, ast(INT_MIN)))
  special2 = ast_and(ast_eq(b, ast(-1)), ast_eq(a, ast(INT_MIN)))
  return z3expr(ast_and(overflow, underflow, special1, special2))

@checker_decorator("divided-by-zero")
def check_divided_by_zero(d):
  return z3expr(ast_eq(d, ast(0)))

def ast(o):
  if hasattr(o, '_ast'):
    return o._ast()
  if isinstance(o, bool):
    return ast_const_bool(o)
  if isinstance(o, int):
    return ast_const_int(o)
  raise Exception("Trying to make an AST out of %s %s" % (o, type(o)))

def value(o):
  if isinstance(o, concolic_int):
    return o._v()
  elif isinstance(o, int):
    return o
  raise Exception("Trying to extract a value out of %s %s" % (o, type(o)))

def concolic_bool(sym, v):
  ## Python claims that 'bool' is not an acceptable base type,
  ## so it seems difficult to subclass bool.  Luckily, bool has
  ## only two possible values, so whenever we get a concolic
  ## bool, add its value to the constraint.
  add_pc(ast_eq(sym, ast(v)))
  return v

class concolic_int(int):
  def __new__(cls, sym, v):
    self = super(concolic_int, cls).__new__(cls, v)
    self.__v = v
    self.__ast = sym
    return self

  def __eq__(self, o):
    if not isinstance(o, int):
      return False
    res = (self.__v == value(o))
    return concolic_bool(ast_eq(self.__ast, ast(o)), res)

  def __ne__(self, o):
    return not self.__eq__(o)

  def __cmp__(self, o):
    res = self.__v.__cmp__(o)
    if concolic_bool(ast_lt(self.__ast, ast(o)), res < 0):
      return -1
    if concolic_bool(ast_gt(self.__ast, ast(o)), res > 0):
      return 1
    return 0

  def __lt__(self, o):
    res = self.__v < o
    return concolic_bool(ast_lt(self.__ast, ast(o)), res)

  def __le__(self, o):
    res = self.__v <= o
    return concolic_bool(ast_not(ast_gt(self.__ast, ast(o))), res)

  def __gt__(self, o):
    res = self.__v > o
    return concolic_bool(ast_gt(self.__ast, ast(o)), res)

  def __ge__(self, o):
    res = self.__v >= o
    return concolic_bool(ast_not(ast_lt(self.__ast, ast(o))), res)

  def __add__(self, o):
    if _check_overflow:
      check_add_overflow(self.__ast, ast(o))
    res = self.__v + value(o)
    return concolic_int(ast_plus(self.__ast, ast(o)), res)

  def __radd__(self, o):
    if _check_overflow:
      check_add_overflow(ast(o), self.__ast)
    res = value(o) + self.__v
    return concolic_int(ast_plus(ast(o), self.__ast), res)

  def __sub__(self, o):
    if _check_overflow:
      check_minus_overflow(self.__ast, ast(o))
    res = self.__v - value(o)
    return concolic_int(ast_minus(self.__ast, ast(o)), res)

  def __rsub__(self, o):
    if _check_overflow:
      check_minus_overflow(ast(o), self.__ast)
    res = value(o) - self.__v
    return concolic_int(ast_minus(ast(o), self.__ast), res)

  def __mul__(self, o):
    if _check_overflow:
      check_multiply_overflow(self.__ast, ast(o))
    res = self.__v * value(o)
    return concolic_int(ast_mul(self.__ast, ast(o)), res)

  def __rmul__(self, o):
    if _check_overflow:
      check_multiply_overflow(ast(o), self.__ast)
    res = value(o) * self.__v
    return concolic_int(ast_mul(ast(o), self.__ast), res)

  def __floordiv__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(ast(o))
    res = self.__v // value(o)
    return concolic_int(ast_div(self.__ast, ast(o)), res)

  def __rfloordiv__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(self.__ast)
    res = value(o) // self.__v
    return concolic_int(ast_div(ast(o), self.__ast), res)

  def __mod__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(ast(o))
    res = self.__v % value(o)
    return concolic_int(ast_mod(self.__ast, ast(o)), res)

  def __rmod__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(self.__ast)
    res = value(o) % self.__v
    return concolic_int(ast_mod(ast(o), self.__ast), res)

  def __and__(self, o):
    res = self.__v & value(o)
    return concolic_int(ast_bwand(self.__ast, ast(o)), res)

  def __rand__(self, o):
    res = value(o) & self.__v
    return concolic_int(ast_bwand(ast(o), self.__ast), res)

  def __or__(self, o):
    res = self.__v | value(o)
    return concolic_int(ast_bwor(self.__ast, ast(o)), res)

  def __ror__(self, o):
    res = value(o) | self.__v
    return concolic_int(ast_bwor(ast(o), self.__ast), res)

  def __lshift__(self, o):
    res = self.__v << value(o)
    return concolic_int(ast_lshift(self.__ast, ast(o)), res)

  def __rlshift__(self, o):
    res = value(o) << self.__v
    return concolic_int(ast_lshift(ast(o), self.__ast), res)

  def __rshift__(self, o):
    res = self.__v >> value(o)
    return concolic_int(ast_rshift(self.__ast, ast(o)), res)

  def __rrshift__(self, o):
    res = value(o) >> self.__v
    return concolic_int(ast_rshift(ast(o), self.__ast), res)

  def _v(self):
    return self.__v

  def _ast(self):
    return self.__ast

## create a concolic_int instance.
def mk_int(id):
  global _vals
  return concolic_int(ast_int(id), _vals[id])

def flip_pc(pc):
  assert isinstance(pc, ast_eq)
  return ast_eq(pc.a, ast(not pc.b.b))

def concolic(f, eval_pc=None, exit_on_err=True, debug=False):
  global _vals, _cur_pc

  ## no heuristics if "eval_pc" is None, i.e. all execution paths
  ## are treated equally. Otherwise, path with higher "eval_pc"
  ## outcome will be prioritized for exploration.
  if eval_pc == None:
    eval_pc = lambda _: 0

  ## "checked" is the set of constraints we already sent to Z3 for
  ## checking. Use this to eliminate duplicate paths.
  checked = set()

  ## "crashes" is the list of inputs lead to crash (or more
  ## accurately, raise exceptions).
  crashes = []

  class q_item(object):
    def __init__(self, priority, value):
      self.priority = priority
      self.value = value

    def __lt__(self, o):
      return o.priority.__lt__(self.priority)

  ## pending queue.
  q = queue.PriorityQueue()
  q.put(q_item(0, defaultdict(int)))

  ## number of iterations so far.
  iters = 0

  while not q.empty():
    _vals = q.get().value
    iters += 1

    log("=" * 60)
    log("[%d] %s" % (iters, model_str(_vals)))

    _cur_pc = []
    try:
      f()
    except:
      sys.excepthook(*sys.exc_info())
      if exit_on_err:
        return
      crashes.append(_vals.copy())

    while _cur_pc:
      new_pc = _cur_pc[:-1]
      new_pc.append(flip_pc(_cur_pc[-1]))

      _cur_pc = _cur_pc[:-1]

      new_path = ast_and(*new_pc)
      if new_path in checked:
        continue
      checked.add(new_path)

      solver.push()
      solver.add(z3expr(new_path))

      res = solver.check()
      if res == sat:
        m = solver.model()
        new_vals = defaultdict(int)
        for var in m.decls():
          ## Python integer is signed.
          new_vals[var.name()] = m[var].as_signed_long()
        q.put(q_item(eval_pc(new_pc), new_vals))

        if debug:
          log("%s -> %s" % (map(str, new_pc), new_vals))

      solver.pop()

  return crashes