#!/usr/bin/env python3



import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules
def test_me():
  x, y = mk_int("x"), mk_int("y")
  if x > 0 and y > 0:
    z = 1000 // (x * y)

if __name__ == "__main__":
  enable_divided_by_zero_check()
  concolic(test_me)