#!/usr/bin/env python3


import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
## Example taken from angr paper. Slightly modified.
## (State of) The Art of War: Offensive Techniques in Binary Analysis.


# Add parent directory to path to allow importing modules
def test_me():
  num = mk_int("num")
  value = 0
  bs = []

  for i in range(5):
    bs.append(mk_int("byte[%d]" % i))

  if num == 111111111:
    value = bs[num]

  if num < 100 and num % 15 == 2 and num % 11 == 6:
    ## only num=17 is a solution
    value = bs[num]

  count = 0
  for b in bs:
    if b == 25:
      count = count + 1

  if count >= 4 and count <= 5:
    value = bs[count * 20]

if __name__ == "__main__":
  crashes = concolic(test_me, exit_on_err=False)

  print("found %d crashes" % len(crashes))
  for c in crashes:
    print(model_str(c))