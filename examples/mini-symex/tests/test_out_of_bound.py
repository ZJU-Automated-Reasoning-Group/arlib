#!/usr/bin/env python3



import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules
@checker_decorator("out-of-bound")
def check_out_of_bound(lst, i):
  if isinstance(i, concolic_int):
    l = ast_lt(ast(i), ast(0))
    h = ast_gt(ast(i), ast(len(lst) - 1))
    return z3expr(ast_or(l, h))
  return None

class mylist(list):
  def __getitem__(self, i):
    check_out_of_bound(self, i)
    return list.__getitem__(self, i)

  def __setitem__(self, i, y):
    check_out_of_bound(self, i)
    return list.__setitem__(self, i, v)

## we can notice there is a wild bug in below code.
## we cannot find the out-of-bound if applying symbolic execution directly,
## but can expose that bug if we hook list.__xxxitem__.
def test_me_fail():
  xs = list()
  xs.extend([1, 2, 3, 4])
  i = mk_int("i")
  if i >= 0 and i <= len(xs):
    y = xs[i]

def test_me():
  xs = mylist()
  xs.extend([1, 2, 3, 4])
  i = mk_int("i")
  if i >= 0 and i <= len(xs):
    y = xs[i]

if __name__ == "__main__":
  log('test_me_fail')
  concolic(test_me_fail)

  for i in range(3):
    log('')

  log('test_me')
  concolic(test_me)