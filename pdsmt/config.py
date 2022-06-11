# coding: utf-8
from enum import Enum


class SolverResult(Enum):
    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    BUG = 3


class ApproximationType(Enum):
    OVER_APPROX = 0
    UNDER_APPRO = 1
    EXACT = 2


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


m_parallel_mode = ParallelMode.USE_MULIT_PROCESSING
m_theory_solver_incremental_type = TheorySolverIncrementalType.NO_INCREMENTAL
m_theory_solver_refinement_strategy = TheorySolverRefinementStrategy.USE_MODEL

m_boolean_sampler_strategy = BooleanSamplerStrategy.NO_UNIFORM

m_smt_solver_bin = f"/Users/peisenyao/Work/z3/build/z3 -in"
# bin_cmd = f"/Users/prism/Work/cvc5/build/bin/cvc5 -q --produce-models -i"
