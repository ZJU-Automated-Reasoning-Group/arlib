from symast import *
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
  if isinstance(o, float):
    return ast_const_float(o)
  if isinstance(o, str):
    return ast_const_str(o)
  if isinstance(o, list):
    # For lists, we create a constant array representation
    # This is a simplified approach - a more complete implementation would
    # track symbolic constraints for each element
    return ast_const_array(o)
  if isinstance(o, dict):
    # For dictionaries, we create a constant dictionary representation
    return ast_const_dict(o)
  raise Exception("Trying to make an AST out of %s %s" % (o, type(o)))

def value(o):
  if isinstance(o, concolic_int):
    return o._v()
  elif isinstance(o, concolic_float):
    return o._v()
  elif isinstance(o, concolic_str):
    return o._v()
  elif isinstance(o, concolic_list):
    return o._v()
  elif isinstance(o, concolic_dict):
    return o._v()
  elif isinstance(o, (int, float, str, list, dict)):
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
  # Handle different types of path constraints
  if isinstance(pc, ast_eq):
    # For equality constraints, negate the boolean value
    if isinstance(pc.b, ast_const_bool):
      return ast_eq(pc.a, ast(not pc.b.b))
    else:
      # For non-boolean equality, use not equal
      return ast_not(pc)
  elif isinstance(pc, ast_lt):
    # For less than, flip to greater than or equal
    return ast_not(ast_lt(pc.a, pc.b))
  elif isinstance(pc, ast_gt):
    # For greater than, flip to less than or equal
    return ast_not(ast_gt(pc.a, pc.b))
  else:
    # For other constraints, use logical negation
    return ast_not(pc)

def concolic(f, eval_pc=None, exit_on_err=True, debug=False, max_states=None, max_paths=None):
  """
  Run concolic execution on function f.
  
  Args:
    f: Function to execute
    eval_pc: Function to evaluate path constraints for prioritization.
             If None, all paths are treated equally.
    exit_on_err: Whether to exit on error/exception
    debug: Whether to print debug information
    max_states: Maximum number of states to explore (None for unlimited)
    max_paths: Maximum number of paths to explore (None for unlimited)
    
  Returns:
    A tuple of (crashes, stats) where:
    - crashes: List of inputs that lead to crashes/exceptions
    - stats: Dictionary containing exploration statistics:
      - states_explored: Number of states explored
      - paths_explored: Number of paths explored
      - max_queue_size: Maximum size of the exploration queue
      - solver_calls: Number of calls to the Z3 solver
      - solver_sat: Number of satisfiable results from the solver
      - solver_unsat: Number of unsatisfiable results from the solver
  """
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
  
  ## number of paths explored so far
  paths_explored = 0

  ## Statistics to track exploration
  stats = {
    "states_explored": 0,
    "paths_explored": 0,
    "max_queue_size": 1,
    "solver_calls": 0,
    "solver_sat": 0,
    "solver_unsat": 0
  }

  while not q.empty():
    # Check if we've reached the maximum number of states
    if max_states is not None and stats["states_explored"] >= max_states:
      log(f"Reached maximum number of states: {max_states}")
      break
      
    # Check if we've reached the maximum number of paths
    if max_paths is not None and stats["paths_explored"] >= max_paths:
      log(f"Reached maximum number of paths: {max_paths}")
      break
      
    _vals = q.get().value
    iters += 1
    stats["states_explored"] += 1

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
      stats["paths_explored"] += 1
      
      # Check if we've reached the maximum number of paths
      if max_paths is not None and stats["paths_explored"] >= max_paths:
        log(f"Reached maximum number of paths: {max_paths}")
        break

      solver.push()
      solver.add(z3expr(new_path))
      stats["solver_calls"] += 1

      res = solver.check()
      if res == sat:
        stats["solver_sat"] += 1
        m = solver.model()
        new_vals = defaultdict(int)
        for var in m.decls():
          ## Handle different Z3 number types
          val = m[var]
          if z3.is_int(val) or z3.is_bv(val):
            new_vals[var.name()] = val.as_signed_long()
          elif z3.is_real(val):
            # For floating point values (represented as RatNumRef)
            num_str = str(val)
            if '/' in num_str:
              # Handle rational numbers like "5/2"
              num, denom = num_str.split('/')
              new_vals[var.name()] = float(num) / float(denom)
            else:
              # Handle decimal numbers
              new_vals[var.name()] = float(num_str)
          else:
            # Default fallback for other types
            new_vals[var.name()] = 0
            
        q.put(q_item(eval_pc(new_pc), new_vals))
        stats["max_queue_size"] = max(stats["max_queue_size"], q.qsize())

        if debug:
          log("%s -> %s" % (map(str, new_pc), new_vals))
      else:
        stats["solver_unsat"] += 1

      solver.pop()

  # Print exploration statistics
  if debug:
    log("=" * 60)
    log("Exploration Statistics:")
    log(f"States explored: {stats['states_explored']}")
    log(f"Paths explored: {stats['paths_explored']}")
    log(f"Maximum queue size: {stats['max_queue_size']}")
    log(f"Solver calls: {stats['solver_calls']}")
    log(f"Solver SAT results: {stats['solver_sat']}")
    log(f"Solver UNSAT results: {stats['solver_unsat']}")
    log("=" * 60)

  return crashes, stats

class concolic_float(float):
  def __new__(cls, sym, v):
    self = super(concolic_float, cls).__new__(cls, v)
    self.__v = v
    self.__ast = sym
    return self

  def __eq__(self, o):
    if not isinstance(o, (float, int)):
      return False
    res = (self.__v == value(o))
    return concolic_bool(ast_eq(self.__ast, ast(o)), res)

  def __ne__(self, o):
    return not self.__eq__(o)

  def __lt__(self, o):
    res = self.__v < value(o)
    return concolic_bool(ast_lt(self.__ast, ast(o)), res)

  def __le__(self, o):
    res = self.__v <= value(o)
    return concolic_bool(ast_not(ast_gt(self.__ast, ast(o))), res)

  def __gt__(self, o):
    res = self.__v > value(o)
    return concolic_bool(ast_gt(self.__ast, ast(o)), res)

  def __ge__(self, o):
    res = self.__v >= value(o)
    return concolic_bool(ast_not(ast_lt(self.__ast, ast(o))), res)

  def __add__(self, o):
    res = self.__v + value(o)
    return concolic_float(ast_plus(self.__ast, ast(o)), res)

  def __radd__(self, o):
    res = value(o) + self.__v
    return concolic_float(ast_plus(ast(o), self.__ast), res)

  def __sub__(self, o):
    res = self.__v - value(o)
    return concolic_float(ast_minus(self.__ast, ast(o)), res)

  def __rsub__(self, o):
    res = value(o) - self.__v
    return concolic_float(ast_minus(ast(o), self.__ast), res)

  def __mul__(self, o):
    res = self.__v * value(o)
    return concolic_float(ast_mul(self.__ast, ast(o)), res)

  def __rmul__(self, o):
    res = value(o) * self.__v
    return concolic_float(ast_mul(ast(o), self.__ast), res)

  def __truediv__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(ast(o))
    res = self.__v / value(o)
    return concolic_float(ast_div(self.__ast, ast(o)), res)

  def __rtruediv__(self, o):
    if _check_divided_by_zero:
      check_divided_by_zero(self.__ast)
    res = value(o) / self.__v
    return concolic_float(ast_div(ast(o), self.__ast), res)

  def _v(self):
    return self.__v

  def _ast(self):
    return self.__ast

class concolic_str(str):
  def __new__(cls, sym, v):
    self = super(concolic_str, cls).__new__(cls, v)
    self.__v = v
    self.__ast = sym
    return self

  def __eq__(self, o):
    if not isinstance(o, str):
      return False
    res = (self.__v == value(o))
    return concolic_bool(ast_eq(self.__ast, ast(o)), res)

  def __ne__(self, o):
    return not self.__eq__(o)

  def __add__(self, o):
    # Convert to string if not already
    if not isinstance(o, str):
      o = str(o)
    res = self.__v + value(o)
    return concolic_str(ast_str_concat(self.__ast, ast(o)), res)

  def __radd__(self, o):
    # Convert to string if not already
    if not isinstance(o, str):
      o = str(o)
    res = value(o) + self.__v
    return concolic_str(ast_str_concat(ast(o), self.__ast), res)

  def __contains__(self, o):
    if not isinstance(o, str):
      o = str(o)
    res = o in self.__v
    return concolic_bool(ast_str_contains(self.__ast, ast(o)), res)

  def startswith(self, prefix, start=0, end=None):
    res = self.__v.startswith(value(prefix), start, end)
    return concolic_bool(ast_str_prefixof(ast(prefix), self.__ast), res)

  def endswith(self, suffix, start=0, end=None):
    res = self.__v.endswith(value(suffix), start, end)
    return concolic_bool(ast_str_suffixof(ast(suffix), self.__ast), res)

  def __len__(self):
    res = len(self.__v)
    # Create a symbolic representation of the length
    sym_len = ast_str_length(self.__ast)
    return concolic_int(sym_len, res)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      # Handle slices - this is simplified and doesn't fully track symbolic constraints
      start = 0 if idx.start is None else value(idx.start)
      stop = len(self.__v) if idx.stop is None else value(idx.stop)
      step = 1 if idx.step is None else value(idx.step)
      res = self.__v[start:stop:step]
      # Create a new symbolic string for the result
      sym_result = ast_str(f"{self.__ast.id}_slice_{start}_{stop}_{step}")
      return concolic_str(sym_result, res)
    else:
      # Handle single character access
      idx_val = value(idx)
      if idx_val < 0 or idx_val >= len(self.__v):
        # Out of bounds check
        add_pc(ast_and(ast_ge(ast(idx), ast(0)), ast_lt(ast(idx), ast(len(self.__v)))))
        raise IndexError("String index out of range")
      res = self.__v[idx_val]
      sym_result = ast_str_at(self.__ast, ast(idx))
      return concolic_str(sym_result, res)

  def _v(self):
    return self.__v

  def _ast(self):
    return self.__ast

class concolic_list(list):
  def __init__(self, sym, v):
    super(concolic_list, self).__init__(v)
    self.__v = list(v)  # Make a copy to avoid external modifications
    self.__ast = sym
    
  def __eq__(self, o):
    if not isinstance(o, list):
      return False
    if len(self.__v) != len(o):
      return False
    # Check each element
    for i in range(len(self.__v)):
      if not self.__v[i] == value(o[i]):
        return False
    return True
    
  def __ne__(self, o):
    return not self.__eq__(o)
    
  def __len__(self):
    return len(self.__v)
    
  def __getitem__(self, idx):
    if isinstance(idx, slice):
      # Handle slices - simplified approach
      start = 0 if idx.start is None else value(idx.start)
      stop = len(self.__v) if idx.stop is None else value(idx.stop)
      step = 1 if idx.step is None else value(idx.step)
      res = self.__v[start:stop:step]
      # Create a new symbolic array for the result
      sym_result = ast_array(f"{self.__ast.id}_slice_{start}_{stop}_{step}", len(res))
      return concolic_list(sym_result, res)
    else:
      # Handle single element access
      idx_val = value(idx)
      if idx_val < 0 or idx_val >= len(self.__v):
        # Out of bounds check
        add_pc(ast_and(ast_ge(ast(idx), ast(0)), ast_lt(ast(idx), ast(len(self.__v)))))
        raise IndexError("List index out of range")
      res = self.__v[idx_val]
      
      # If the result is a primitive type that we support, wrap it
      if isinstance(res, int):
        sym_result = ast_select(self.__ast, ast(idx))
        return concolic_int(sym_result, res)
      elif isinstance(res, float):
        sym_result = ast_select(self.__ast, ast(idx))
        return concolic_float(sym_result, res)
      elif isinstance(res, str):
        sym_result = ast_select(self.__ast, ast(idx))
        return concolic_str(sym_result, res)
      else:
        # For other types, just return the concrete value
        return res
        
  def __setitem__(self, idx, val):
    idx_val = value(idx)
    if idx_val < 0 or idx_val >= len(self.__v):
      # Out of bounds check
      add_pc(ast_and(ast_ge(ast(idx), ast(0)), ast_lt(ast(idx), ast(len(self.__v)))))
      raise IndexError("List index out of range")
      
    # Update concrete value
    self.__v[idx_val] = value(val)
    
    # Update symbolic representation
    self.__ast = ast_store(self.__ast, ast(idx), ast(val))
    
  def append(self, val):
    # Add to concrete list
    self.__v.append(value(val))
    
    # For symbolic representation, we create a new array with the appended value
    # This is a simplified approach - a more complete implementation would
    # track symbolic constraints for dynamic resizing
    new_size = len(self.__v)
    new_ast = ast_array(f"{self.__ast.id}_append_{new_size}", new_size)
    
    # Copy existing elements
    for i in range(new_size - 1):
      new_ast = ast_store(new_ast, ast(i), ast_select(self.__ast, ast(i)))
      
    # Add new element
    new_ast = ast_store(new_ast, ast(new_size - 1), ast(val))
    
    self.__ast = new_ast
    
  def _v(self):
    return self.__v
    
  def _ast(self):
    return self.__ast

## create concolic instances for different types
def mk_float(id):
  global _vals
  return concolic_float(ast_float(id), float(_vals[id]))

def mk_str(id):
  global _vals
  if id not in _vals:
    _vals[id] = ""
  elif not isinstance(_vals[id], str):
    # Convert non-string values to strings
    _vals[id] = str(_vals[id])
  return concolic_str(ast_str(id), _vals[id])

def mk_list(id, size=0):
  global _vals
  if id not in _vals:
    _vals[id] = [0] * size
  elif not isinstance(_vals[id], list):
    # Convert non-list values to lists
    if isinstance(_vals[id], int):
      _vals[id] = [_vals[id]] * size
    else:
      _vals[id] = list(str(_vals[id]))
  return concolic_list(ast_array(id, len(_vals[id])), _vals[id])

class concolic_dict(dict):
  def __init__(self, sym, v):
    super(concolic_dict, self).__init__(v)
    self.__v = dict(v)  # Make a copy to avoid external modifications
    self.__ast = sym
    
  def __eq__(self, o):
    if not isinstance(o, dict):
      return False
    if len(self.__v) != len(o):
      return False
    # Check each key-value pair
    for k in self.__v:
      if k not in o or not self.__v[k] == value(o[k]):
        return False
    return True
    
  def __ne__(self, o):
    return not self.__eq__(o)
    
  def __len__(self):
    return len(self.__v)
    
  def __getitem__(self, key):
    if key not in self.__v:
      # Key not found
      add_pc(ast_dict_contains(self.__ast, ast(key)))
      raise KeyError(key)
      
    res = self.__v[key]
    
    # If the result is a primitive type that we support, wrap it
    if isinstance(res, int):
      # Use hash of key as the index
      key_hash = hash(key) % (2**31)
      sym_result = ast_select(self.__ast, ast(key_hash))
      return concolic_int(sym_result, res)
    elif isinstance(res, float):
      key_hash = hash(key) % (2**31)
      sym_result = ast_select(self.__ast, ast(key_hash))
      return concolic_float(sym_result, res)
    elif isinstance(res, str):
      key_hash = hash(key) % (2**31)
      sym_result = ast_select(self.__ast, ast(key_hash))
      return concolic_str(sym_result, res)
    else:
      # For other types, just return the concrete value
      return res
      
  def __setitem__(self, key, val):
    # Update concrete value
    self.__v[key] = value(val)
    
    # Update symbolic representation
    # Use hash of key as the index
    key_hash = hash(key) % (2**31)
    self.__ast = ast_store(self.__ast, ast(key_hash), ast(val))
    
  def __contains__(self, key):
    res = key in self.__v
    return concolic_bool(ast_dict_contains(self.__ast, ast(key)), res)
    
  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default
    
  def _v(self):
    return self.__v
    
  def _ast(self):
    return self.__ast

def mk_dict(id):
  global _vals
  if id not in _vals:
    _vals[id] = {}
  elif not isinstance(_vals[id], dict):
    # Convert non-dict values to dicts
    _vals[id] = {0: _vals[id]}
  
  # Ensure we have at least one key-value pair for testing
  if len(_vals[id]) == 0:
    _vals[id] = {"key1": 10}
    
  return concolic_dict(ast_dict(id), _vals[id])

## Enhanced control flow support

def symbolic_range(start, stop=None, step=1):
  """
  A symbolic-aware range function that handles symbolic bounds
  
  Args:
    start: Start value (or stop if stop is None)
    stop: Stop value (or None)
    step: Step value
    
  Returns:
    A generator that yields values with appropriate path constraints
  """
  if stop is None:
    stop = start
    start = 0
  
  # Handle concrete case
  if not isinstance(start, concolic_int) and not isinstance(stop, concolic_int):
    for i in range(start, stop, step):
      yield i
    return
  
  # For symbolic bounds, we need to be careful
  concrete_start = value(start)
  concrete_stop = value(stop)
  
  # Add path constraint that the loop will execute at least once
  add_pc(ast_lt(ast(concrete_start), ast(concrete_stop)))
  
  # Return a generator that yields values and updates path constraints
  i = concrete_start
  while i < concrete_stop:
    yield concolic_int(ast_plus(ast(start), ast(i - concrete_start)), i)
    i += step
    # Add constraint that we're still in bounds
    add_pc(ast_lt(ast(i), ast(stop)))

def symbolic_try_except(try_block, except_blocks=None, finally_block=None):
  """
  A symbolic-aware try-except-finally construct
  
  Args:
    try_block: Function to execute in try block
    except_blocks: List of (exception_type, handler_function) pairs
    finally_block: Function to execute in finally block
    
  Returns:
    Result of try_block or matching exception handler
  """
  global _cur_pc
  
  # Default empty except_blocks
  if except_blocks is None:
    except_blocks = []
  
  # Save current path constraints
  old_pc = _cur_pc.copy()
  
  try:
    # Try to execute the try block
    result = try_block()
    return result
  except Exception as e:
    # An exception was raised, find matching handler
    for exc_type, handler in except_blocks:
      if isinstance(e, exc_type):
        # Found matching handler
        return handler(e)
    # No matching handler, re-raise
    raise
  finally:
    if finally_block:
      finally_block()

## Object-oriented support

class concolic_object(object):
  """Base class for concolic objects"""
  def __init__(self, sym, concrete_obj):
    self.__sym = sym
    self.__concrete = concrete_obj
  
  def __getattr__(self, name):
    # Get attribute from concrete object
    concrete_attr = getattr(self.__concrete, name)
    
    # If it's a method, wrap it to maintain concolic state
    if callable(concrete_attr):
      def wrapped_method(*args, **kwargs):
        # Convert concolic args to concrete
        concrete_args = [value(arg) for arg in args]
        concrete_kwargs = {k: value(v) for k, v in kwargs.items()}
        
        # Call concrete method
        result = concrete_attr(*concrete_args, **concrete_kwargs)
        
        # Wrap result if needed
        if isinstance(result, (int, float, str, list, dict)):
          # Create appropriate concolic wrapper based on type
          if isinstance(result, int):
            sym_result = ast_int(f"{self.__sym.id}_{name}_result")
            return concolic_int(sym_result, result)
          elif isinstance(result, float):
            sym_result = ast_float(f"{self.__sym.id}_{name}_result")
            return concolic_float(sym_result, result)
          elif isinstance(result, str):
            sym_result = ast_str(f"{self.__sym.id}_{name}_result")
            return concolic_str(sym_result, result)
          elif isinstance(result, list):
            sym_result = ast_array(f"{self.__sym.id}_{name}_result", len(result))
            return concolic_list(sym_result, result)
          elif isinstance(result, dict):
            sym_result = ast_dict(f"{self.__sym.id}_{name}_result")
            return concolic_dict(sym_result, result)
        return result
      
      return wrapped_method
    
    return concrete_attr
  
  def _v(self):
    return self.__concrete
  
  def _ast(self):
    return self.__sym

def mk_object(id, cls, *args, **kwargs):
  """
  Create a concolic object instance
  
  Args:
    id: Symbolic identifier
    cls: Class to instantiate
    *args, **kwargs: Arguments to pass to the constructor
    
  Returns:
    A concolic_object wrapping an instance of cls
  """
  # Convert concolic args to concrete
  concrete_args = [value(arg) for arg in args]
  concrete_kwargs = {k: value(v) for k, v in kwargs.items()}
  
  # Create concrete instance
  concrete_obj = cls(*concrete_args, **concrete_kwargs)
  
  # Create symbolic representation
  sym = ast_int(id)  # Use int as a simple symbolic representation
  
  return concolic_object(sym, concrete_obj)

## Improved path exploration strategies

def concolic_coverage_guided(f, max_iterations=100, max_states=None, max_paths=None):
  """
  Run concolic execution with coverage-guided path exploration
  
  Prioritizes paths that increase code coverage
  
  Args:
    f: Function to execute
    max_iterations: Maximum number of iterations
    max_states: Maximum number of states to explore (None for unlimited)
    max_paths: Maximum number of paths to explore (None for unlimited)
    
  Returns:
    Tuple of (crashes, stats) where crashes is a list of inputs that lead to crashes
    and stats is a dictionary of exploration statistics
  """
  global _vals, _cur_pc
  
  # Track coverage
  covered_branches = set()
  
  # Define evaluation function for path constraints
  def eval_pc(pc):
    # Calculate a score based on how many new branches this path might cover
    score = 0
    for constraint in pc:
      # Create a unique identifier for this branch
      branch_id = hash(constraint)
      if branch_id not in covered_branches:
        score += 1
    return score
  
  # Run concolic execution with our evaluation function
  return concolic(f, eval_pc=eval_pc, exit_on_err=False, debug=True, 
                 max_states=max_states, max_paths=max_paths)

def concolic_directed(f, target_line, max_iterations=100, max_states=None, max_paths=None):
  """
  Run concolic execution directed towards a specific line of code
  
  Uses static analysis to guide path exploration towards target_line
  
  Args:
    f: Function to execute
    target_line: Line number to target
    max_iterations: Maximum number of iterations
    max_states: Maximum number of states to explore (None for unlimited)
    max_paths: Maximum number of paths to explore (None for unlimited)
    
  Returns:
    Tuple of (crashes, stats) where crashes is a list of inputs that lead to crashes
    and stats is a dictionary of exploration statistics
  """
  # This would require static analysis of the function's control flow graph
  # to determine which branches lead towards the target line
  
  # For simplicity, we'll use a placeholder implementation
  def eval_pc(pc):
    # In a real implementation, we would analyze which branches
    # are more likely to lead to target_line
    return 0
  
  return concolic(f, eval_pc=eval_pc, exit_on_err=False, debug=True,
                 max_states=max_states, max_paths=max_paths)

## Automatic instrumentation

def instrument_function(func):
  """
  Automatically instrument a function for concolic execution
  
  Uses Python's introspection to identify parameters and local variables
  
  Args:
    func: Function to instrument
    
  Returns:
    Wrapper function that uses concolic execution
  """
  import inspect
  
  # Get function signature
  sig = inspect.signature(func)
  
  # Create a wrapper function
  def wrapper(*args, **kwargs):
    # Create symbolic variables for parameters
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    symbolic_args = {}
    for name, value in bound_args.arguments.items():
      if isinstance(value, int):
        symbolic_args[name] = mk_int(name)
      elif isinstance(value, float):
        symbolic_args[name] = mk_float(name)
      elif isinstance(value, str):
        symbolic_args[name] = mk_str(name)
      elif isinstance(value, list):
        symbolic_args[name] = mk_list(name, len(value))
      elif isinstance(value, dict):
        symbolic_args[name] = mk_dict(name)
      else:
        symbolic_args[name] = value
    
    # Call the original function with symbolic arguments
    return func(**symbolic_args)
  
  return wrapper

def concolic_exec(func, *args, **kwargs):
  """
  Execute a function with concolic execution
  
  Args:
    func: Function to execute
    *args, **kwargs: Arguments to pass to the function
    
  Returns:
    Tuple of (crashes, stats) where crashes is a list of inputs that lead to crashes
    and stats is a dictionary of exploration statistics
  """
  # Extract concolic execution parameters if provided
  max_states = kwargs.pop('max_states', None)
  max_paths = kwargs.pop('max_paths', None)
  
  # Create instrumented version of the function
  instrumented_func = instrument_function(func)
  
  # Create a wrapper that calls the instrumented function with the provided args
  def wrapper():
    return instrumented_func(*args, **kwargs)
  
  # Run concolic execution on the wrapper
  return concolic(wrapper, exit_on_err=False, debug=True, 
                 max_states=max_states, max_paths=max_paths)

