# -*- coding: utf-8 -*-

from z3 import *
from ast import *
from utils import *
from posix_ipc import Semaphore, O_CREAT
import os
import sys
import atexit

N_CPUS = 4

## Use semaphore to limit the number of concurrent processes,
## we allow N_CPUS processes running simultaneously.
sem = Semaphore("/fork_sem", O_CREAT, 0o644, N_CPUS)
sem.unlink()
sem.acquire()

def on_exit():
  sem.release()
  ## reap all zombies.
  try:
    while True:
      os.waitpid(0)
  except:
    pass
  log("exit")

atexit.register(on_exit)

def ast(o):
  if hasattr(o, '_ast'):
    return o._ast()
  if isinstance(o, bool):
    return ast_const_bool(o)
  if isinstance(o, int):
    return ast_const_int(o)
  raise Exception("Trying to make an AST out of %s %s" % (o, type(o)))

def symbolic_bool(sym):
  pid = os.fork()
  ## even we limit the number of processes running simultaneously,
  ## we still need to fork one process which stays in blocked state
  ## until the semaphore is active... so, memory usage is a shitty issue.
  if pid:  # parent
    solver.add(z3expr(ast_eq(sym, ast(True))))
    r = True
    log("assume (%s)" % str(sym))
  else:  # child
    sem.acquire()
    solver.add(z3expr(ast_eq(sym, ast(False))))
    r = False
    log("assume ¬(%s)" % str(sym))
  if solver.check() != sat:
    log("unreachable")
    sys.exit(0)
  return r

class symbolic_int(object):
  def __init__(self, sym):
    self.__ast = sym

  def __eq__(self, o):
    if isinstance(o, int) or isinstance(o, symbolic_int):
      return symbolic_bool(ast_eq(self.__ast, ast(o)))
    return False

  def __ne__(self, o):
    if isinstance(o, int) or isinstance(o, symbolic_int):
      return symbolic_bool(ast_not(ast_eq(self.__ast, ast(o))))
    return True

  def __cmp__(self, o):
    if symbolic_bool(ast_lt(self.__ast, ast(o))):
      return -1
    if symbolic_bool(ast_gt(self.__ast, ast(o))):
      return 1
    return 0

  def __lt__(self, o):
    return symbolic_bool(ast_lt(self.__ast, ast(o)))

  def __le__(self, o):
    return symbolic_bool(ast_not(ast_gt(self.__ast, ast(o))))

  def __gt__(self, o):
    return symbolic_bool(ast_gt(self.__ast, ast(o)))

  def __ge__(self, o):
    return symbolic_bool(ast_not(ast_lt(self.__ast, ast(o))))

  def __add__(self, o):
    return symbolic_int(ast_plus(self.__ast, ast(o)))

  def __radd__(self, o):
    return symbolic_int(ast_plus(ast(o), self.__ast))

  def __sub__(self, o):
    return symbolic_int(ast_minus(self.__ast, ast(o)))

  def __rsub__(self, o):
    return symbolic_int(ast_minus(ast(o), self.__ast))

  def __mul__(self, o):
    return symbolic_int(ast_mul(self.__ast, ast(o)))

  def __rmul__(self, o):
    return symbolic_int(ast_mul(ast(o), self.__ast))

  def __floordiv__(self, o):
    return symbolic_int(ast_div(self.__ast, ast(o)))

  def __rfloordiv__(self, o):
    return symbolic_int(ast_div(ast(o), self.__ast))

  def __mod__(self, o):
    return symbolic_int(ast_mod(self.__ast, ast(o)))

  def __rmod__(self, o):
    return symbolic_int(ast_mod(ast(o), self.__ast))

  def __and__(self, o):
    return symbolic_int(ast_bwand(self.__ast, ast(o)))

  def __rand__(self, o):
    return symbolic_int(ast_bwand(ast(o), self.__ast))

  def __or__(self, o):
    return symbolic_int(ast_bwor(self.__ast, ast(o)))

  def __ror__(self, o):
    return symbolic_int(ast_bwor(ast(o), self.__ast))

  def __lshift__(self, o):
    return symbolic_int(ast_lshift(self.__ast, ast(o)))

  def __rlshift__(self, o):
    return symbolic_int(ast_lshift(ast(o), self.__ast))

  def __rshift__(self, o):
    return symbolic_int(ast_rshift(self.__ast, ast(o)))

  def __rrshift__(self, o):
    return symbolic_int(ast_rshift(ast(o), self.__ast))

  def _ast(self):
    return self.__ast

def mk_int(id):
  return symbolic_int(ast_int(id))