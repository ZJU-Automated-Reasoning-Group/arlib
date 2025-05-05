"""
Configurations for the CDCL-based SMT engine
"""

from enum import Enum
import os


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
    If the theory solver finds a Boolean model is T-inconsistency,
     how should it help refine the Boolean abstraction
    """
    USE_MODEL = 0  # just return the spurious Boolean model
    USE_ANY_UNSAT_CORE = 1  # an arbitrary unsat core
    USE_MIN_UNSAT_CORE = 2  # minimal unsat core


class BooleanSamplerStrategy(Enum):
    """How to sample Boolean models"""
    NO_UNIFORM = 0  # just randomly generate a few Boolean models
    UNIGEN = 1  # use unigen


# Set the Parallel Mode
m_parallel_mode = ParallelMode.USE_MULIT_PROCESSING

# Set the Theory Solver Incremental Type
m_theory_solver_incremental_type = TheorySolverIncrementalType.NO_INCREMENTAL

# Set the Theory Solver Refinement Strategy
m_theory_solver_refinement_strategy = TheorySolverRefinementStrategy.USE_MODEL

# Set the Initial Abstraction Strategy
m_init_abstraction = InitAbstractionStrategy.ATOM

# Set the Boolean Sampler Strategy
m_boolean_sampler_strategy = BooleanSamplerStrategy.NO_UNIFORM

# Solver configuration
MAX_SOLVER_TIME_SEC = 300
MIN_SOLVER_TIME_SEC = 30
MAX_UNKNOWN_TIMES = 1

# Query splitting
MAX_QUERY_SIZE = 5000
ALLOW_AUTO_SPLITTING = True

# Formula rewriting
CONVERT_TO_BOOL_WHEN_POSSIBLE = True
REMOVE_VAR_ARGS = True

# Boolean abstraction
SIMPLIFY = True
MIN_BOOLEAN_SUBS = 0
MIN_SKIP_SUBST = 0
CNF_SIMPLIFY = True

# Portfolio
ALLOW_PORTFOLIO = True
PORTFOLIO_SELECTION_SIZE = 2

# Portfolio settings
PAR_VARS = None
PAR_CNTS = None
PAR_MIN_LEVEL = 1

# Sampling
NUM_SAMPLES = 10
BOOL_SOLVER_RANDOM_SEED = 1

# T-consistency Checking
PARALLEL_T_CHECKING = True
MAX_T_CHECKING_PROCESSES = 0  # 0 means use number of CPUs

# SMT Query Logging
LOG_SMT_QUERIES = True  # Set to False to disable logging
SMT_LOG_DIR = os.path.join(os.getcwd(), "smt_query_logs")  # Default log directory
SMT_LOG_QUERY_CONTENT = True  # Log the full content of SMT queries
SMT_LOG_QUERY_RESULTS = True  # Log the results of SMT queries
SMT_LOG_ASSUMPTIONS = True    # Log assumptions for check-sat-assuming calls

# Boolean Solver
SOLVER_ENGINE = 'glucose4'  # Default SAT solver engine

# Proof Production
PRODUCE_PROOF = False
PROOF_FILE = ""
