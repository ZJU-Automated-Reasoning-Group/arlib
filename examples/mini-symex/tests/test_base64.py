#!/usr/bin/env python3


import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
# Add parent directory to path to allow importing modules
## THIS IMPLEMENTATION OF `byte_encode` DOES NOT WORK!
## We will lose the symbolic part of concolic values.
##
## BYTES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
##
## def byte_encode(x):
##   return ord(BYTES[x])

def byte_encode(x):
  if x < 26:
    return x + ord('A')
  elif x < 52:
    return x - 26 + ord('a')
  elif x < 62:
    return x - 52 + ord('0')
  elif x < 63:
    return ord('+')
  elif x < 64:
    return ord('/')
  raise Exception("Never be here")

def base64_encode(data):
  length = len(data)
  output = [ord('=')] * ((length + 2) // 3 * 4)
  ptr, i = 0, 0

  while len(data) >= 3:
    first, second, third = data[:3]
    data = data[3:]

    n = first << 16 | second << 8 | third
    output[ptr], ptr = byte_encode((n >> 18) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 12) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 6) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 0) & 63), ptr + 1

  if len(data) == 1:
    n = data[0] << 16
    output[ptr], ptr = byte_encode((n >> 18) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 12) & 63), ptr + 1
  elif len(data) == 2:
    n = data[0] << 16 | data[1] << 8
    output[ptr], ptr = byte_encode((n >> 18) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 12) & 63), ptr + 1
    output[ptr], ptr = byte_encode((n >> 6) & 63), ptr + 1

  return output

def test_me():
  data = [mk_int(chr(i + ord('a'))) for i in range(6)]

  ## assume all input characters are visible.
  def assume_ok(data):
    for x in data:
      if x < 0x20 or x > 0x7e:
        return False
    return True

  def bytes_equal(xs, ys):
    if len(xs) != len(ys):
      return False
    for x, y in zip(xs, ys):
      if x != ord(y):
        return False
    return True

  if assume_ok(data):
    if bytes_equal(base64_encode(data), "aDFiYWJ5"):
      ## 679 rounds of iteration to trigger this assertion.
      ## [679] a = 0x68, b = 0x31, c = 0x62, d = 0x61, e = 0x62, f = 0x79
      assert False, "reach me"

## heuristic to prioritize the path leads to `assert False`.
##
## this heuristic is based on some observations from the code. To expose
## `assert False`, we need to pass all checks in bytes_equal, so we give
## path with equality a positive score, while inequality a high negative
## score.
def eval_pc(pc):
  goodness = 0
  for i in pc:
    assert isinstance(i, ast_eq)
    i, v = i.a, i.b
    if isinstance(i, ast_eq):
      if v.b:
        goodness += 100
      else:
        goodness -= 1000
    else:
      goodness -= 1
  return goodness

if __name__ == "__main__":
  return
  # to slow, so we don't run it.
  # concolic(test_me, eval_pc=eval_pc)