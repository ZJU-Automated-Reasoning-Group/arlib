#!/usr/bin/env python3

from concolic import *

def test_me():
  x = mk_int("x")
  y = (x >> 1) * 2
  z = y + 1
  k = z + 1

if __name__ == "__main__":
  enable_overflow_check()
  concolic(test_me)