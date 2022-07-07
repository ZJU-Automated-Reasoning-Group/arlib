# coding: utf-8
"""
Some global configurations
TODO: consider multiple sources/approaches of configuring the values, e.g.,
  - The file global_params/paths.py
  - Command line options
"""
from .global_params.paths import z3_exec, cvc5_exec
from .utils import ParallelMode, TheorySolverRefinementStrategy, \
    TheorySolverIncrementalType, BooleanSamplerStrategy, InitAbstractionStrategy

m_parallel_mode = ParallelMode.USE_MULIT_PROCESSING
m_theory_solver_incremental_type = TheorySolverIncrementalType.NO_INCREMENTAL
m_theory_solver_refinement_strategy = TheorySolverRefinementStrategy.USE_MODEL

m_init_abstraction = InitAbstractionStrategy.ATOM
# m_init_abstraction = InitAbstractionStrategy.CLAUSE

m_boolean_sampler_strategy = BooleanSamplerStrategy.NO_UNIFORM

m_smt_solver_bin = z3_exec + " -in"
m_cvc5_solver_bin = cvc5_exec + " -q -i"
