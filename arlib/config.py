# coding: utf-8
"""
Some global configurations
TODO: consider multiple sources/approaches of configuring the values, e.g.,
  - The file global_params/paths.py
  - Command line options
"""
z3_exec = ""
cvc5_exec = ""
m_smt_solver_bin = z3_exec + " -in"
m_cvc5_solver_bin = cvc5_exec + " -q -i"
