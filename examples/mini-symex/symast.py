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

# New constant types
class ast_const_float(ast_base):
  def __init__(self, f):
    self.f = f

  def __eq__(self, o):
    if not isinstance(o, ast_const_float):
      return False
    return self.f == o.f

  def __hash__(self):
    return hash(self.f)

  def _z3expr(self):
    return self.f

class ast_const_str(ast_base):
  def __init__(self, s):
    self.s = s

  def __eq__(self, o):
    if not isinstance(o, ast_const_str):
      return False
    return self.s == o.s

  def __hash__(self):
    return hash(self.s)

  def _z3expr(self):
    return StringVal(self.s)

class ast_const_array(ast_base):
  def __init__(self, arr):
    self.arr = arr

  def __eq__(self, o):
    if not isinstance(o, ast_const_array):
      return False
    if len(self.arr) != len(o.arr):
      return False
    return all(a == b for a, b in zip(self.arr, o.arr))

  def __hash__(self):
    return hash(tuple(self.arr))

  def _z3expr(self):
    # Create a Z3 array constant with the values from self.arr
    # Start with an empty array
    result = z3.K(z3.IntSort(), 0)
    # Add each element
    for i, val in enumerate(self.arr):
      if hasattr(val, '_z3expr'):
        z3_val = val._z3expr()
      else:
        # Convert Python value to appropriate Z3 value
        if isinstance(val, bool):
          z3_val = val
        elif isinstance(val, int):
          z3_val = val
        elif isinstance(val, float):
          z3_val = val
        elif isinstance(val, str):
          z3_val = StringVal(val)
        else:
          z3_val = 0  # Default for unsupported types
      result = z3.Store(result, i, z3_val)
    return result

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

# New symbolic types
class ast_float(ast_base):
  def __init__(self, id):
    self.id = id

  def __eq__(self, o):
    if not isinstance(o, ast_float):
      return False
    return self.id == o.id

  def __hash__(self):
    return hash(self.id)

  def _z3expr(self):
    return z3.Real(self.id)

class ast_str(ast_base):
  def __init__(self, id):
    self.id = id

  def __eq__(self, o):
    if not isinstance(o, ast_str):
      return False
    return self.id == o.id

  def __hash__(self):
    return hash(self.id)

  def _z3expr(self):
    return z3.String(self.id)

class ast_array(ast_base):
  def __init__(self, id, size):
    self.id = id
    self.size = size

  def __eq__(self, o):
    if not isinstance(o, ast_array):
      return False
    return self.id == o.id and self.size == o.size

  def __hash__(self):
    return hash((self.id, self.size))

  def _z3expr(self):
    return z3.Array(self.id, z3.IntSort(), z3.IntSort())

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

class ast_bwxor(ast_binop):  # bitwise xor
  def _z3expr(self):
    return z3expr(self.a) ^ z3expr(self.b)

class ast_bwnot(ast_unop):
  def _z3expr(self):
    return ~z3expr(self.a)

# String operations
class ast_str_concat(ast_binop):
  def _z3expr(self):
    return z3.Concat(z3expr(self.a), z3expr(self.b))

class ast_str_contains(ast_binop):
  def _z3expr(self):
    return z3.Contains(z3expr(self.a), z3expr(self.b))

class ast_str_prefixof(ast_binop):
  def _z3expr(self):
    return z3.PrefixOf(z3expr(self.a), z3expr(self.b))

class ast_str_suffixof(ast_binop):
  def _z3expr(self):
    return z3.SuffixOf(z3expr(self.a), z3expr(self.b))

class ast_str_at(ast_binop):
  def _z3expr(self):
    # Use the at method of SeqRef instead of the non-existent At function
    a_expr = z3expr(self.a)
    b_expr = z3expr(self.b)
    return a_expr.at(b_expr)

class ast_str_length(ast_unop):
  def _z3expr(self):
    return z3.Length(z3expr(self.a))

# Array operations
class ast_select(ast_binop):
  def _z3expr(self):
    return z3.Select(z3expr(self.a), z3expr(self.b))

class ast_store(ast_func_apply):
  def __init__(self, array, index, value):
    super(ast_store, self).__init__(array, index, value)
  
  def _z3expr(self):
    return z3.Store(z3expr(self.args[0]), z3expr(self.args[1]), z3expr(self.args[2]))

# Dictionary operations
class ast_dict(ast_base):
  def __init__(self, id):
    self.id = id

  def __eq__(self, o):
    if not isinstance(o, ast_dict):
      return False
    return self.id == o.id

  def __hash__(self):
    return hash(self.id)

  def _z3expr(self):
    # We model dictionaries as arrays from keys to values
    # For simplicity, we'll use integers as keys (hash of the actual key)
    return z3.Array(self.id, z3.IntSort(), z3.IntSort())

class ast_const_dict(ast_base):
  def __init__(self, d):
    self.d = d

  def __eq__(self, o):
    if not isinstance(o, ast_const_dict):
      return False
    return self.d == o.d

  def __hash__(self):
    # Convert dict to a hashable representation
    return hash(tuple(sorted(self.d.items())))

  def _z3expr(self):
    # Create a Z3 array constant with the values from self.d
    # Start with an empty array
    result = z3.K(z3.IntSort(), 0)
    # Add each element
    for k, v in self.d.items():
      # Use hash of key as the index
      key_hash = hash(k) % (2**31)
      
      if hasattr(v, '_z3expr'):
        z3_val = v._z3expr()
      else:
        # Convert Python value to appropriate Z3 value
        if isinstance(v, bool):
          z3_val = v
        elif isinstance(v, int):
          z3_val = v
        elif isinstance(v, float):
          z3_val = v
        elif isinstance(v, str):
          z3_val = StringVal(v)
        else:
          z3_val = 0  # Default for unsupported types
      
      result = z3.Store(result, key_hash, z3_val)
    return result

class ast_dict_contains(ast_binop):
  def _z3expr(self):
    # Check if key is in dictionary
    # We use the hash of the key as the index
    key_hash = hash(z3expr(self.b)) % (2**31)
    # This is a simplified approach - in reality, we'd need to check for hash collisions
    return z3.Select(z3expr(self.a), key_hash) != 0
  
