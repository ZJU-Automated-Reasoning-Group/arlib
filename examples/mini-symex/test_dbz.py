#!/usr/bin/env python3

from concolic import *

def test_me():
  x, y = mk_int("x"), mk_int("y")
  if x > 0 and y > 0:
    z = 1000 // (x * y)

if __name__ == "__main__":
  enable_divided_by_zero_check()
  concolic(test_me)