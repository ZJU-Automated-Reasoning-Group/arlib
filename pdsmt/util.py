# coding: utf-8
import re
from enum import Enum


class SolverResult(Enum):
    UNSAT = -1
    UNKNOWN = 0
    SAT = 1
    ERROR = 2


class InitAbstractionStrategy(Enum):
    """
    First, perform simplification and CNF transformation
    Then, build a Boolean abstraction
    """
    ATOM = -1 # the traditional way: map each atom to a Boolean variable
    CLAUSE = 0 # map each clause to a Boolean variable
    RAND_CUBE = 1 # construct a set of random cubes? (not clear yet)


class ParallelMode(Enum):
    USE_MULIT_PROCESSING = 0
    USE_THREADING = 1
    USE_MPI = 2


class TheorySolverIncrementalType(Enum):
    NO_INCREMENTAL = 0  # do not use incremental solving
    PUSH_POP = 1  # use push/pop
    ASSUMPTIONS = 2  # use assumption literals


class TheorySolverRefinementStrategy(Enum):
    USE_MODEL = 0  # just return the spurious Boolean model
    USE_ANY_UNSAT_CORE = 1  # an arbitrary unsat core
    USE_MIN_UNSAT_CORE = 2  # minimal unsat core


class BooleanSamplerStrategy(Enum):
    NO_UNIFORM = 0  # just randomly generate a few Boolean models
    UNIGEN = 1  # use unigen


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
