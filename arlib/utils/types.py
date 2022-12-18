# coding: utf-8
"""
Common types for different components of arlib
"""
import re
from enum import Enum


class SolverResult(Enum):
    UNSAT = -1
    UNKNOWN = 0
    SAT = 1
    ERROR = 2


RE_GET_EXPR_VALUE_ALL = re.compile(
    # r"\(([a-zA-Z0-9_]*)[ \n\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false)\)"
    r"\((p@[0-9]*)[ \n\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false)\)"
)


def convert_value(v):
    """
    For converting SMT-LIB models to Python values
    TODO: we may need to deal with other types of variables, e.g., int, real, string, etc.
    """
    r = None
    if v == "true":
        r = True
    elif v == "false":
        r = False
    elif v.startswith("#b"):
        r = int(v[2:], 2)
    elif v.startswith("#x"):
        r = int(v[2:], 16)
    elif v.startswith("_ bv"):
        r = int(v[len("_ bv"): -len(" 256")], 10)
    elif v.startswith("(_ bv"):
        v = v[len("(_ bv"):]
        r = int(v[: v.find(" ")], 10)

    assert r is not None
    return r
