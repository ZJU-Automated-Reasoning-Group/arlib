from __future__ import print_function
import atexit, multiprocessing, ctypes
import sys, os
import z3
from pygments import highlight
from pygments.formatters import get_formatter_by_name
from pygments.lexers import get_lexer_by_name
from traceback import format_exception

solver = z3.Solver()

lock = multiprocessing.Lock()

def log(s):
  # "atomic" print; less concern about performance
  with lock:
    print("[%s] %s" % (os.getpid(), s), file=sys.stderr)

def model_str(self):
  if isinstance(self, dict):
    return ", ".join([
        "%s = %s" % (str(k), ("0x%x" % ctypes.c_uint(v).value))
        for k, v in sorted(self.items())
    ])
  if isinstance(self, z3.ModelRef):
    return ", ".join([
        "%s = %s" % (k,
                     ("0x%x" % ctypes.c_uint(self[k].as_signed_long()).value))
        for k in sorted(self.decls(), key=str)
    ])

setattr(z3.ModelRef, "__str__", model_str)
setattr(z3.ModelRef, "__repr__", model_str)

def on_excepthook(exctype, value, traceback):
  msg = ''.join(format_exception(exctype, value, traceback)).strip()
  # print input/model
  if solver.check() == z3.sat:
    model = solver.model()
    msg = "%s: %s" % (msg, str(model))
  # highlight output
  lexer = get_lexer_by_name("pytb", stripall=True)
  formatter = get_formatter_by_name("terminal256")
  msg = highlight(msg, lexer, formatter)
  log(msg)

if sys.stderr.isatty():
  sys.excepthook = on_excepthook