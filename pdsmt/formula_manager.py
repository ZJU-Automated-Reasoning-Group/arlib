# coding: utf-8
import logging

logger = logging.getLogger(__name__)


class BooleanFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # s-expression of the signature
        self.smt2_init_cnt = ""  # initial cnt in SMT2 (without "assert")

        self.numeric_clauses = []  # initial cnt in numerical clauses
        self.bool_vars_name = []  # p@0, p@1, ...
        self.num2vars = {}  # 0 -> p@0, ...
        self.vars2num = {}  # p@0 -> 1


class TheoryFormulaManager(object):

    def __init__(self):
        self.smt2_signature = []  # variables
        self.smt2_init_cnt = ""