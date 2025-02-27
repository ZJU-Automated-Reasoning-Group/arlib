#!/usr/bin/env python3

from concolic import *

def test_me():
  x, y = mk_int("x"), mk_int("y")
  z = x << 1 | 1
  if z == y:
    if y == x + 0x11:
      assert False, "reach me"

if __name__ == "__main__":
  concolic(test_me, debug=True, exit_on_err=False)