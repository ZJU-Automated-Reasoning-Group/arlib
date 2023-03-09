"""
Configurations for the CDCL-based SMT engine
"""

from enum import Enum


class InitAbstractionStrategy(Enum):
    """
    This abstraction strategy for the CDCL(T) SMT engine.

    First, perform simplification and CNF transformation
    Second, build a Boolean abstraction.
    """
    ATOM = -1  # the traditional way: map each atom to a Boolean variable
    CLAUSE = 0  # map each clause to a Boolean variable
    RAND_CUBE = 1  # construct a set of random cubes? (not clear yet)


class ParallelMode(Enum):
    """
    Parallel solving mode
    """
    USE_MULIT_PROCESSING = 0
    USE_THREADING = 1
    USE_MPI = 2


class TheorySolverIncrementalType(Enum):
    """
    Strategy for incremental solving
    """
    NO_INCREMENTAL = 0  # do not use incremental solving
    PUSH_POP = 1  # use push/pop
    ASSUMPTIONS = 2  # use assumption literals


class TheorySolverRefinementStrategy(Enum):
    """
    """
    USE_MODEL = 0  # just return the spurious Boolean model
    USE_ANY_UNSAT_CORE = 1  # an arbitrary unsat core
    USE_MIN_UNSAT_CORE = 2  # minimal unsat core


class BooleanSamplerStrategy(Enum):
    NO_UNIFORM = 0  # just randomly generate a few Boolean models
    UNIGEN = 1  # use unigen


m_parallel_mode = ParallelMode.USE_MULIT_PROCESSING
m_theory_solver_incremental_type = TheorySolverIncrementalType.NO_INCREMENTAL
m_theory_solver_refinement_strategy = TheorySolverRefinementStrategy.USE_MODEL

m_init_abstraction = InitAbstractionStrategy.ATOM
# m_init_abstraction = InitAbstractionStrategy.CLAUSE

m_boolean_sampler_strategy = BooleanSamplerStrategy.NO_UNIFORM
