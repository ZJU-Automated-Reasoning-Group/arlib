# coding: utf-8
"""
Some global configurations
TODO: consider multiple sources/approaches of configuring the values, e.g.,
  - The file global_params/paths.py
  - Command line options
"""
from arlib.global_params import global_config

m_smt_solver_bin = global_config.z3_exec + " -in"
m_cvc5_solver_bin =  global_config.cvc5_exec + " -q -i"
