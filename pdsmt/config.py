# coding: utf-8
from .util import ParallelMode, TheorySolverRefinementStrategy, \
    TheorySolverIncrementalType, BooleanSamplerStrategy, InitAbstractionStrategy

m_parallel_mode = ParallelMode.USE_MULIT_PROCESSING
m_theory_solver_incremental_type = TheorySolverIncrementalType.NO_INCREMENTAL
m_theory_solver_refinement_strategy = TheorySolverRefinementStrategy.USE_MODEL

m_init_abstraction = InitAbstractionStrategy.ATOM
# m_init_abstraction = InitAbstractionStrategy.CLAUSE

m_boolean_sampler_strategy = BooleanSamplerStrategy.NO_UNIFORM

m_smt_solver_bin = f"/Users/prism/Work/z3bin/bin/z3 -in"
# bin_cmd = f"/Users/prism/Work/cvc5/build/bin/cvc5 -q --produce-models -i"
