# coding: utf-8
"""
Some global configurations
"""
from arlib.global_params import global_config

m_smt_solver_bin = global_config.get_solver_path("z3") + " -in"
m_cvc5_solver_bin =  global_config.get_solver_path("cvc5") + " -q -i"
