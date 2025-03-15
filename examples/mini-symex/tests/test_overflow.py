#!/usr/bin/env python3



import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules
def test_me():
  x = mk_int("x")
  y = (x >> 1) * 2
  z = y + 1
  k = z + 1

if __name__ == "__main__":
  enable_overflow_check()
  concolic(test_me)