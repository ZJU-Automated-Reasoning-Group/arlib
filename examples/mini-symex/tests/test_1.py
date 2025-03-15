#!/usr/bin/env python3



import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules
def test_me():
  x, y = mk_int("x"), mk_int("y")
  z = x << 1 | 1
  if z == y:
    if y == x + 0x11:
      assert False, "reach me"

if __name__ == "__main__":
  concolic(test_me, debug=True, exit_on_err=False)