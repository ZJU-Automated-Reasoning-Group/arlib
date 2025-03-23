# coding: utf-8
"""
Randomly generating SMT-LIB2 formulas

This file is the one usd by the "smtfuzz" Python library..
You may install the tool by
    pip install smtfuzz
"""
import argparse
import itertools
import math
import random
import signal
import string
import sys
from collections import OrderedDict

m_set_logic = True  # print (set-logic LO) or not
# if random.random() < 0.1: m_set_logic = False
# tmp test; should be true

m_strict_cnf = False

m_reset_assert = False  # (reset-assertions)
m_reset_cmd = '(reset-assertions)'
if random.random() < 0.2:
    m_reset_cmd = '(reset)'

m_test_fp64 = False  # default Float32
if random.random() < 0.5:
    m_test_fp64 = True
m_fp_rounding_mode = "random"
if random.random() < 0.8:  # use fixed?
    fp_round_pp = random.random()
    if fp_round_pp < 0.2:
        m_fp_rounding_mode = "RNE"
    elif fp_round_pp < 0.4:
        m_fp_rounding_mode = "RNA"
    elif fp_round_pp < 0.6:
        m_fp_rounding_mode = "RTP"
    elif fp_round_pp < 0.8:
        m_fp_rounding_mode = "RTN"
    else:
        m_fp_rounding_mode = "RTZ"

m_test_iand = False
m_test_eqrange = False

# For generator
# '''
m_quantifier_rate_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_or_and_rate_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_exists_forall_rate_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_assert_or_create_new_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_create_exp_rate_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_create_bool_rate_swam = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
m_push_pop_rate_swam = [0.05, 0.1, 0.15, 0.2, 0.25]

m_declare_new_var_swam = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40]

m_max_smt_rate = random.uniform(1.05, 1.66) - 1
# '''
m_quantifier_rate = random.choice(m_quantifier_rate_swam)
m_or_and_rate = random.choice(m_or_and_rate_swam)
m_exists_forall_rate = random.choice(m_exists_forall_rate_swam)
m_assert_or_create_new = random.choice(m_assert_or_create_new_swam)
m_create_exp_rate = random.choice(m_create_exp_rate_swam)
m_create_bool_rate = random.choice(m_create_bool_rate_swam)
m_push_pop_rate = random.choice(m_push_pop_rate_swam)
m_declare_new_var_rate = random.choice(m_declare_new_var_swam)
# '''
'''
m_quantifier_rate = random.uniform(1.05, 1.66) - 1
m_or_and_rate = random.uniform(1.05, 1.66) - 1
m_exists_forall_rate = random.uniform(1.05, 1.66) - 1
m_assert_or_create_new = random.uniform(1.05, 1.66) - 1
m_create_exp_rate = random.uniform(1.05, 1.66) - 1
m_create_bool_rate = random.uniform(1.05, 1.66) - 1
m_push_pop_rate = random.uniform(1.05, 1.66) - 1
m_declare_new_var_rate = random.uniform(1.05, 1.66) - 1
'''

m_init_var_size = 20  # default 20

m_use_swam_bv = False  # selectively reduce the number of bv-operations
# if random.random() < 0.5: m_use_swam_bv = True
m_use_bv_concat_repeat = True
# if random.random() < 0.33: m_use_bv_concat_repeat = False

m_use_swam_fp = False  # selectively reduce the number of fp-operations
if random.random() < 0.33:
    m_use_swam_fp = True

m_use_fancy_qterm = False  # create fancy quantified assertions
if random.random() < 0.66:
    m_use_fancy_qterm = True

m_use_ite = False  # ite(b s s)
if random.random() < 0.22:
    m_use_ite = True

# Advanced features
m_test_smt_opt = False  # (maximize x)
m_test_smt_opt_fancy_term = False
if random.random() < 0.33:
    m_test_smt_opt_fancy_term = True

m_test_max_smt = False  # (assert-soft xx)
m_test_qe = False  # quantifier elimination
m_test_unsat_core = False  # unsat core
m_test_interpolant = False  # interpolant
m_test_yices_itp = False  # get-unsat-model-interpolant in yices2
m_test_yices_core = False  #

m_assert_id = 0  # id for naming assertions in unsat_core
m_all_assertions = []
m_backtrack_points = []
n_push = 0
n_pop = 0
m_fancy_push = False
# if random.random() < 0.25: m_fancy_push = True

m_test_abduction = False  # abduction (CVC4 only)
m_test_cvc_itp = False  # cvc4 interpolant

m_test_proof = False  # proof generation

m_test_z3_interpolant = False  # z3 ITP
m_test_z3_simplify = False
m_test_z3_tactic = False

m_test_named_assert = False  # just test named assertions

m_test_pure_sat = False  # SAT
m_test_qbf = False  # QBF
m_test_max_sat = False  # maxsatm_test_allsat

m_test_set_bapa = False  # Set and/or BAPA
m_test_str_set_bapa = False  # Set of strings
m_test_bag_bapa = False  # Bag and/or BAPA
m_test_set_comprehension = False
m_test_set_eq = False  # seteq

# String-related o
m_test_string = False  # Test string
m_test_string_lia = False
m_test_z3str3 = False
if random.random() < 0.33:
    m_test_z3str3 = True
m_test_seq = False

m_test_string_substr = False
if random.random() < 0.33:
    m_test_string_substr = True

m_test_string_re = False
if random.random() < 0.5:
    m_test_string_re = True  # TODO: many re  operations not added

m_test_string_replace = False
if random.random() < 0.33:
    m_test_string_replace = True

m_test_string_unicode = False
if random.random() < 0.22:
    m_test_string_unicode = True

m_use_swam_str = False  # selectively reduce the number of str-operations
if random.random() < 0.5:
    m_use_swam_str = True

"""
FP related
"""
m_test_fp = False  # Test FP
m_test_fp_no_num = False  # FP, but do not create any num.
m_test_fp_lra = False

m_test_seplog = False  # Separation logic

"""Datalog/CHC"""
m_test_datalog_chc = False  # Datalog and CHC
m_test_datalog_chc_logic = "int"  # underlying theory
m_test_datalog_chc_var_bound = 3  # max. number of variable in each rule
m_test_datalog_chc_nonlinear = False  # generate non-linear term of not
m_test_datalog_chc_as_tactic = False  # test CHC with "horn" tactic

"""Recursive function"""
m_test_recfun = False  # recursive function
m_test_recfun_logic = "int"

m_test_my_uf = False  # uninterpreted functions

m_test_bvint = False  # BV and INT

"""SMTInterpol"""
m_test_smtinterpol = False

"""CVC4"""
m_test_cvc4 = False  # cvc4 mode
m_noinc_mode = False

m_test_ufc = False  # UF with card

# Boolector
m_test_boolector = False
m_test_stp = False  # share the code of new_Array wth Boolector

# Yices2
m_test_yices = False

# eldarica
m_test_eldarica = False

# diff
m_test_diff = False
m_test_diff_core = False

# Global info.
m_global_logic = ''
m_global_strategy = ''

# option fuzz mode
m_optionmode = 'full'  # none, basic, full

# should be QF_SLIA?
# QF_IDL, IDL
qf_int_logic_options = ['QF_UFIDL', 'QF_IDL', 'QF_S', 'QF_SLIA', 'QF_UFSLIA', 'QF_SNIA', 'QF_NIA', 'QF_LIA', 'QF_ANIA',
                        'QF_ALIA', 'QF_AUFNIA', 'QF_AUFLIA', 'QF_UFNIA', 'QF_UFLIA']
q_int_logic_options = [
    'ALIA',
    'ANIA',
    'LIA',
    'NIA',
    'UFLIA',
    'UFNIA',
    'AUFLIA',
    'AUFNIA']
int_logic_options = qf_int_logic_options + q_int_logic_options

# QF_RDL, RDL
qf_real_logic_options = ['QF_UFRDL', 'QF_RDL', 'QF_ANRA', 'QF_ALRA', 'QF_FPLRA', 'QF_UFLRA', 'QF_NRA',
                         'QF_LRA', 'QF_UFNRA', 'QF_AUFNRA', 'QF_AUFLRA']
q_real_logic_options = [
    'ANRA',
    'ALRA',
    'LRA',
    'NRA',
    'UFLRA',
    'UFNRA',
    'AUFLRA',
    'AUFNRA']

lira_logics = [
    'QF_LIRA',
    'QF_SLIRA',
    'LIRA',
    'QF_ALIRA',
    'ALIRA',
    'QF_UFLIRA',
    'UFLIRA',
    'QF_AUFLIRA',
    'AUFLIRA']
nira_logics = [
    'QF_NIRA',
    'NIRA',
    'QF_ANIRA',
    'ANIRA',
    'QF_UFNIRA',
    'UFNIRA',
    'QF_AUFNIRA',
    'AUFNIRA']
qf_ira_logics = ['QF_LIRA', 'QF_SLIRA', 'QF_ALIRA', 'QF_UFLIRA', 'QF_AUFLIRA', 'QF_NIRA', 'QF_ANIRA', 'QF_UFNIRA',
                 'QF_AUFNIRA']

# QF_SNIA is not included
lia_logics = ['QF_SLIA', 'SEQ', 'QF_UFSLIA', 'QF_UFSLIA', 'IDL', 'QF_IDL', 'QF_UFIDL', 'LIA', 'UFLIA', 'ALIA', 'AUFLIA',
              'QF_LIA', 'QF_UFLIA', 'QF_ALIA', 'QF_AUFLIA']
lra_logics = ['QF_RDL', 'QF_UFRDL', 'RDL', 'QF_FPLRA', 'FPLRA', 'LRA', 'QF_LRA', 'UFLRA', 'QF_UFLRA', 'AUFLRA',
              'QF_AUFLRA']

lia_logics += lira_logics
lra_logics += lira_logics

# ANRA, ALRA??
real_logic_options = qf_real_logic_options + q_real_logic_options

qf_bv_logic_options = ['QF_BV', 'QF_UFBV', 'QF_ABV', 'QF_AUFBV']
q_bv_logic_options = ['BV', 'UFBV', 'ABV', 'AUFBV']
bv_logic_options = qf_bv_logic_options + q_bv_logic_options

qf_logic_options = qf_int_logic_options + \
                   qf_real_logic_options + qf_bv_logic_options
qf_logic_options.append('BOOL')
qf_logic_options.append('QF_UF')
q_logic_options = q_int_logic_options + \
                  q_real_logic_options + q_bv_logic_options
q_logic_options.append('QBF')
q_logic_options.append('UF')

string_logic_options = ['QF_S', 'QF_SLIA', 'QF_SNIA', 'QF_SLIRA', 'QF_UFSLIA']

total_logic_options = qf_logic_options + q_logic_options
total_logic_options += string_logic_options


class Op:
    def __init__(self, node, expr):
        self.expr = expr
        self.node = node

    def __repr__(self):
        return '({} {})'.format(self.node, self.expr)

    def __eq__(self, other):
        return isinstance(
            other, Op) and self.expr == other.expr and self.node == other.node

    def __hash__(self):
        return hash((self.expr, self.node))

    def get_node_ty(self):
        return self.node

    def set_node_ty(self, newnode):
        self.node = newnode


class Ite_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Regular_Op(Op):  # regular exp
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Quantifier_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Bool_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class USort_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Int_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Real_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class FP_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Set_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Bag_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Seplog_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class String_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Seq_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class BV_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Arr_Op(Op):
    def __init__(self, node, expr):
        Op.__init__(self, node, expr)


class Var:
    def __init__(self, sort, n):
        self.sort = sort
        self.n = n

    def __repr__(self):
        return str(self.sort) + str(self.n)


class Var_Bool(Var):
    def __init__(self, n):
        Var.__init__(self, 'v', n)

    def __eq__(self, other):
        return isinstance(other, Var_Bool) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_Int(Var):
    def __init__(self, n):
        Var.__init__(self, 'i', n)

    def __eq__(self, other):
        return isinstance(other, Var_Int) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_Real(Var):
    def __init__(self, n):
        Var.__init__(self, 'r', n)

    def __eq__(self, other):
        return isinstance(other, Var_Real) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_FP(Var):
    def __init__(self, n, fp_sort=None):
        Var.__init__(self, 'fpv', n)
        self.fp_sort = fp_sort or FP()  # Default to standard Float32 if not specified

    def __repr__(self):
        eb = self.fp_sort.eb
        sb = self.fp_sort.sb
        if eb == 8 and sb == 24:
            return f"fpv{self.n}"
        elif eb == 11 and sb == 53:
            return f"fpv{self.n}_64"
        else:
            return f"fpv{self.n}_{eb}_{sb}"

    def __eq__(self, other):
        return (isinstance(other, Var_FP) and 
                self.n == other.n and 
                self.fp_sort == other.fp_sort)

    def __hash__(self):
        return hash((self.sort, self.n, self.fp_sort))


class Var_Set(Var):
    def __init__(self, n):
        Var.__init__(self, 'st', n)

    def __eq__(self, other):
        return isinstance(other, Var_Set) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_Bag(Var):
    def __init__(self, n):
        Var.__init__(self, 'bag', n)

    def __eq__(self, other):
        return isinstance(other, Var_Bag) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_String(Var):
    def __init__(self, n):
        Var.__init__(self, 'str', n)

    def __eq__(self, other):
        return isinstance(other, Var_String) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_Seq(Var):
    def __init__(self, n):
        Var.__init__(self, 'seq', n)

    def __eq__(self, other):
        return isinstance(other, Var_String) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_UnIntSort(Var):
    def __init__(self, sort, n):
        Var.__init__(self, sort, n)

    def __repr__(self):
        return '{}-{}'.format(self.sort, self.n)

    def __eq__(self, other):
        return isinstance(
            other, Var_UnIntSort) and self.n == other.n and self.sort == other.sort

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_BV(Var):
    def __init__(self, sort, n):
        Var.__init__(self, sort, n)

    def __repr__(self):
        return 'bv_{}-{}'.format(self.sort, self.n)

    def __eq__(self, other):
        return isinstance(
            other, Var_BV) and self.n == other.n and self.sort == other.sort

    def __hash__(self):
        return hash((self.sort, self.n))


class Var_Arr(Var):
    def __init__(self, sort_index, sort_element, n):
        Var.__init__(self, sort_index, n)
        self.sort_element = sort_element

    def __repr__(self):
        return 'arr-{}_{}-{}'.format(hash(self.sort),
                                     hash(self.sort_element), self.n)

    def __eq__(self, other):
        return isinstance(other,
                          Var_Arr) and self.n == other.n and self.sort == other.sort and \
            self.sort_element == other.sort_element

    def __hash__(self):
        return hash((self.sort, self.sort_element, self.n))


class Var_Quant(Var):
    def __init__(self, n):
        Var.__init__(self, 'q', n)

    def __eq__(self, other):
        return isinstance(other, Var_Quant) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class Sort:
    def __init__(self, sort):
        self.sort = sort

    def __repr__(self):
        return str(self.sort)


class Bool(Sort):
    def __init__(self):
        Sort.__init__(self, 'Bool')

    def __eq__(self, other):
        return isinstance(other, Bool)

    def __hash__(self):
        return hash(self.sort)


class BoolVar(Sort):
    def __init__(self):
        Sort.__init__(self, 'Bool')

    def __eq__(self, other):
        return isinstance(other, BoolVar)

    def __hash__(self):
        return hash(self.sort)


class Regular(Sort):
    def __init__(self):
        Sort.__init__(self, 're')

    def __eq__(self, other):
        return isinstance(other, Regular)

    def __hash__(self):
        return hash(self.sort)


class Quantifier(Sort):
    def __init__(self):
        Sort.__init__(self, 'qu')

    def __eq__(self, other):
        return isinstance(other, Quantifier)

    def __hash__(self):
        return hash(self.sort)


class Int(Sort):
    def __init__(self):
        Sort.__init__(self, 'Int')

    def __eq__(self, other):
        return isinstance(other, Int)

    def __hash__(self):
        return hash(self.sort)


class IntVar(Sort):
    def __init__(self):
        Sort.__init__(self, 'Int')

    def __eq__(self, other):
        return isinstance(other, IntVar)

    def __hash__(self):
        return hash(self.sort)


class Real(Sort):
    def __init__(self):
        Sort.__init__(self, 'Real')

    def __eq__(self, other):
        return isinstance(other, Real)

    def __hash__(self):
        return hash(self.sort)


class RealVar(Sort):
    def __init__(self):
        Sort.__init__(self, 'Real')

    def __eq__(self, other):
        return isinstance(other, RealVar)

    def __hash__(self):
        return hash(self.sort)


# TODO: allo more FP types
class FP(Sort):
    def __init__(self, eb=8, sb=24):
        # Default is single precision (Float32): 8 bits exponent, 24 bits significand
        self.eb = eb  # exponent bits
        self.sb = sb  # significand bits
        
        # Support both named formats and explicit bit configurations
        if eb == 8 and sb == 24:
            Sort.__init__(self, 'Float32')
        elif eb == 11 and sb == 53:
            Sort.__init__(self, 'Float64')
        elif eb == 15 and sb == 113:
            Sort.__init__(self, 'Float128')
        else:
            # Custom format with explicit bit widths
            Sort.__init__(self, f'(_ FloatingPoint {eb} {sb})')

    def __repr__(self):
        return f'FP({self.eb}, {self.sb})'

    def __eq__(self, other):
        if not isinstance(other, FP):
            return False
        return self.eb == other.eb and self.sb == other.sb

    def __hash__(self):
        return hash((self.sort, self.eb, self.sb))


class Set(Sort):
    def __init__(self):
        Sort.__init__(self, '(Set Int)')

    def __eq__(self, other):
        return isinstance(other, Set)

    def __hash__(self):
        return hash(self.sort)


class Bag(Sort):
    def __init__(self):
        Sort.__init__(self, '(Bag Int)')

    def __eq__(self, other):
        return isinstance(other, Bag)

    def __hash__(self):
        return hash(self.sort)


class String(Sort):
    def __init__(self):
        Sort.__init__(self, '(String)')

    def __eq__(self, other):
        return isinstance(other, String)

    def __hash__(self):
        return hash(self.sort)


class Seq(Sort):
    def __init__(self):
        Sort.__init__(self, '(Seq Int)')

    def __eq__(self, other):
        return isinstance(other, Seq)

    def __hash__(self):
        return hash(self.sort)


class UnIntSort(Sort):
    def __init__(self, n):
        Sort.__init__(self, 'S')
        self.n = n

    def __repr__(self):
        return str(self.sort) + str(self.n)

    def __eq__(self, other):
        return isinstance(other, UnIntSort) and self.n == other.n

    def __hash__(self):
        return hash((self.sort, self.n))


class BV(Sort):
    def __init__(self, w):
        Sort.__init__(self, 'BV')
        self.w = w

    def __repr__(self):
        return '(_ BitVec {})'.format(self.w)

    def __eq__(self, other):
        return isinstance(other, BV) and self.w == other.w

    def __hash__(self):
        return hash((self.sort, self.w))


class Arr(Sort):
    def __init__(self, sort_index, sort_element):
        Sort.__init__(self, 'Arr')
        self.sort_index = sort_index
        self.sort_element = sort_element

    def __repr__(self):
        return '(Array {} {})'.format(self.sort_index, self.sort_element)

    def __eq__(self, other):
        return isinstance(other,
                          Arr) and self.sort_index == other.sort_index and self.sort_element == other.sort_element

    def __hash__(self):
        return hash((self.sort, self.sort_index, self.sort_element))


def random_real():
    y = 0
    if random.random() < 0.8:
        real = str(random.randint(1, 9))
    else:
        real = "0."
        y += 1
    for x in range(random.randint(0, 10)):
        if random.random() < 0.05 and y == 0:
            real += "."
            y += 1
        else:
            real += str(random.randint(0, 9))
    if real[-1] == ".":
        real += "0"
    # NOTE: Fix an important z3 paring error??. A real number should be 2.0,
    # not 2
    if "." not in real:
        real += ".0"
    return real


def get_random_unicode(length):
    try:
        get_char = unichr
    except NameError:
        get_char = chr
    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        get_char(code_point) for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return ''.join(random.choice(alphabet) for _ in range(length))


def random_string(stringLength=6):
    """Generate a random string of fixed length """
    letters = string.ascii_uppercase
    letters += string.ascii_lowercase
    letters += string.digits  # number
    # if random.random() < 0.15: letters += ';.<>+-/_{}=?'
    length = random.randint(1, 15)
    # letters = string.digits + string.ascii_letters + string.punctuation #
    # alread cover uniocode and xx?
    if m_test_string_unicode:
        sp = random.random()
        if sp < 0.45:
            return ''.join(random.choice(letters) for _ in range(length))
        # elif m_test_cvc4:
        elif sp < 0.85:
            bytes_data = get_random_unicode(length).encode('unicode-escape')
            return "".join(map(chr, bytes_data))
        else:
            return ""
    else:
        if random.random() < 0.85:
            return ''.join(random.choice(letters) for _ in range(length))
        else:
            return ""


# TODO !!!!!!!!!! This function is buggy? How to generate random fp???
def random_fp(eb=None, sb=None):
    # If no specific precision is provided, use the global default
    if eb is None and sb is None:
        if m_test_fp64:
            eb, sb = 11, 53  # double precision
        else:
            eb, sb = 8, 24   # single precision
    
    # Generate a random real number
    y = 0
    if random.random() < 0.8:
        real = str(random.randint(1, 9))
    else:
        real = "0."
        y += 1
    
    for x in range(random.randint(0, 10)):
        if random.random() < 0.05 and y == 0:
            real += "."
            y += 1
        else:
            real += str(random.randint(0, 9))
    
    if real[-1] == ".":
        real += "0"
    
    # Ensure it's formatted as a real number
    if "." not in real:
        real += ".0"

    # Select a rounding mode
    if m_fp_rounding_mode == 'random':
        pp = random.random()
        if pp < 0.2:
            rrr = " RNE "  # Round to nearest, ties to even
        elif pp < 0.4:
            rrr = " RNA "  # Round to nearest, ties away from zero
        elif pp < 0.6:
            rrr = " RTP "  # Round toward positive
        elif pp < 0.8:
            rrr = " RTN "  # Round toward negative
        else:
            rrr = " RTZ "  # Round toward zero
    else:
        rrr = " RNE "  # Default rounding mode

    # Format according to SMT-LIB standard
    fp = f"((_ to_fp {eb} {sb}){rrr}{real})"
    
    # Occasionally generate special FP values
    if random.random() < 0.1:
        specials = [
            f"(_ +oo {eb} {sb})",     # positive infinity
            f"(_ -oo {eb} {sb})",     # negative infinity
            f"(_ +zero {eb} {sb})",   # positive zero
            f"(_ -zero {eb} {sb})",   # negative zero
            f"(_ NaN {eb} {sb})"      # Not a Number
        ]
        fp = random.choice(specials)
    
    return fp


def random_BV():
    prob = random.random()
    # num = random.randint(0, 8000)
    num = random.randint(0, 100)
    if prob < 0.33:
        if random.random() < 0.5:
            bv = "#b" + str(bin(num)[2:])
            width = len(str(bin(num)[2:]))
        else:
            bv = "#b0" + str(bin(num)[2:])
            width = len(str(bin(num)[2:])) + 1
    elif prob < 0.66:
        bv = "#x" + str(hex(num)[2:])
        width = len(str(hex(num)[2:])) * 4
    else:
        width = len(str(bin(num)[2:]))
        bv = "(_ bv{} {})".format(num, width)
    return bv, width


def Ratio(lower_bound, upper_bound, ratio):
    n_variables = random.randint(lower_bound, upper_bound)
    n_clauses = math.ceil(ratio * n_variables)
    return n_variables, n_clauses


def find(s, ch):
    return [ii for ii, ltr in enumerate(s) if ltr == ch]


def replace_idx(s, index, replacement):
    return '{}{}{}'.format(s[:index], replacement, s[index + 1:])


def set_options():
    global m_test_proof, m_test_unsat_core

    if m_optionmode == 'none':
        return

    if m_test_proof:
        print("(set-option :produce-proofs true)")
    if m_test_unsat_core or m_test_yices_core:
        print('(set-option :produce-unsat-cores true)')
    if m_test_yices_itp:
        print('(set-option :produce-unsat-model-interpolants true)')
    # if m_test_interpolant and (not m_test_z3_interpolant): print("(set-option :produce-interpolants true)")


def set_logic(logic_choice, option_fuzzing_mode):
    global m_quantifier_rate, m_test_set_bapa, m_test_bag_bapa, m_test_my_uf, m_test_string
    global m_set_logic
    if 'UF' in logic_choice or logic_choice == 'ALL':
        if random.random() < 0.5:
            m_test_my_uf = True
    if 'IDL' in logic_choice or 'RDL' in logic_choice:
        # NOTE!! dangerous operation; I disable all possible non-linear funcs
        global IntBinOp
        global IntNOp
        global RealBinOp
        global RealNOp
        IntBinOp = ["-", "+"]
        IntNOp = ["-", "+"]
        RealBinOp = ["-", "+"]
        RealNOp = ["-", "+"]

    # For testing horn tactic, generate more quantifiers
    if m_test_datalog_chc_as_tactic:
        m_quantifier_rate = 0.33

    a = m_create_exp_rate  # newSort
    b = 0.66  # varUSort
    # c = 1  # bool_from_usort
    c = m_create_bool_rate
    # ni = m_create_exp_rate  # new_int
    ni = m_declare_new_var_rate
    e = m_create_exp_rate  # int_from_int
    # f = m_create_exp_rate  # bool_from_int
    f = m_create_bool_rate
    # g = m_create_exp_rate  # new_real
    g = m_declare_new_var_rate
    h = m_create_exp_rate  # real_from_real
    # m = m_create_exp_rate  # bool_from_real
    m = m_create_bool_rate
    v = m_create_exp_rate  # real_and_int
    # r = m_create_exp_rate  # new_BV
    r = m_declare_new_var_rate
    t = m_create_exp_rate  # BV_from_BV
    # u = m_create_exp_rate  # bool_from_BV
    u = m_create_bool_rate
    gen_arr = m_create_exp_rate  # arrays of any sort
    add_reals = 0
    add_ints = 0
    add_quantifiers = -1

    # not supported by z3: ANIA, 'ANRA', 'QF_ANRA', 'QF_ALRA', 'QF_AUFNRA', "QF_AUFLRA', "UFNRA', 'AUFLRA'??
    # why no AX?  'QF_AX', 'AX'

    if logic_choice == 'ALL':
        if m_set_logic:
            print('(set-logic ALL)')
        set_options()
        add_reals = 1
        add_ints = 1
        if random.random() < 0.33:
            r, t, u = -1, -1, -1  # no BV
        if random.random() < 0.33:
            g, h, m, v = -1, -1, -1, -1
            add_reals = -1
        if random.random() < 0.33:
            ni, e, f, v = -1, -1, -1, -1
            add_ints = -1
        elif random.random() < 0.15:
            m_test_set_bapa = True
        if random.random() < 0.33:
            add_quantifiers = m_quantifier_rate

    elif logic_choice == 'BVINT':
        # if m_set_logic: print('(set-logic ALL)')
        set_options()
        add_ints = 1
        g, h, m, v = -1, -1, -1, -1
        add_reals = -1  # no real
        # ni, e, f, v = -1, -1, -1, -1; add_ints = -1 # keep int
        if random.random() < 0.33:
            add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_ABV':
        if m_set_logic:
            print('(set-logic QF_ABV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate

    elif logic_choice == 'QF_BV':
        if m_set_logic:
            print('(set-logic QF_BV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'QF_AUFBV':
        if m_set_logic:
            print('(set-logic QF_AUFBV)')
        set_options()
        ni, e, f, g, h, m, v, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate

    elif logic_choice == 'QF_NIA':
        if m_set_logic:
            print('(set-logic QF_NIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'QF_ANIA':
        if m_set_logic:
            print('(set-logic QF_ANIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1

    elif logic_choice == 'QF_LIA':
        if m_set_logic:
            print('(set-logic QF_LIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'QF_IDL':
        if m_set_logic:
            print('(set-logic QF_IDL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'QF_UFIDL':
        if m_set_logic:
            print('(set-logic QF_UFIDL)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'IDL':
        if m_set_logic:
            print('(set-logic IDL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_ALIA':
        if m_set_logic:
            print('(set-logic QF_ALIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1

    elif logic_choice == 'QF_UFLIA':
        if m_set_logic:
            print('(set-logic QF_UFLIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'QF_AUFLIA':
        if m_set_logic:
            print('(set-logic QF_AUFLIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1

    elif logic_choice == 'QF_NRA':
        if m_set_logic:
            print('(set-logic QF_NRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_NRAT':
        if m_set_logic:
            print('(set-logic QF_NRAT)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1


    elif logic_choice == 'QF_FP':
        if m_set_logic:
            print('(set-logic QF_FP)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'FP':
        if m_set_logic:
            print('(set-logic FP)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    # FP + Real
    elif logic_choice == 'QF_FPLRA':
        if m_set_logic:
            print('(set-logic QF_FPLRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'FPLRA':
        # if m_set_logic: print('(set-logic FPLRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_LRA':
        if m_set_logic:
            print('(set-logic QF_LRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_RDL':
        if m_set_logic:
            print('(set-logic QF_RDL)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_UFRDL':
        if m_set_logic:
            print('(set-logic QF_UFRDL)')
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_UFLRA':
        if m_set_logic:
            print('(set-logic QF_UFLRA)')
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_ALRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic QF_LRA)')
        else:
            if m_set_logic:
                print('(set-logic QF_AUFLIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1

    elif logic_choice == 'QF_AUFLRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic QF_AUFLRA)')
        else:
            if m_set_logic:
                print('(set-logic QF_AUFLIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1

    elif logic_choice == 'QF_ANRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic QF_ANRA)')
        else:
            if m_set_logic:
                print('(set-logic QF_AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1
        a, b, c = -1, -1, -1

    elif logic_choice == 'QF_AUFNRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic QF_AUFNRA)')
        else:
            if m_set_logic:
                print('(set-logic QF_AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1

    elif logic_choice == 'QF_UF':
        if m_set_logic:
            print('(set-logic QF_UF)')
        set_options()
        ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'QF_UFC':  # uf with card (CVC4 only)
        if m_set_logic:
            print('(set-logic QF_UFC)')
        set_options()
        ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'UFC':  # uf with card (CVC4 only)
        if m_set_logic:
            print('(set-logic UFC)')
        set_options()
        ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_UFBV':
        if m_set_logic:
            print('(set-logic QF_UFBV)')
        set_options()
        ni, e, f, g, h, m, v, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'QF_UFNRA':
        if m_set_logic:
            print('(set-logic QF_UFNRA)')
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1

    elif logic_choice == 'QF_UFNIA':
        if m_set_logic:
            print('(set-logic QF_UFNIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'QF_AUFNIA':
        if m_set_logic:
            print('(set-logic QF_AUFNIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1

    # elif logic_choice == 'QF_AX':
    #    if m_set_logic: print('(set-logic QF_AX)')
    #    set_options(option_fuzzing)
    #    add_reals = -1
    #    add_ints = -1
    elif logic_choice == 'QF_ABV':
        if m_set_logic:
            print('(set-logic QF_ABV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'QF_AUFBV':
        if m_set_logic:
            print('(set-logic QF_AUFBV)')
        set_options()
        ni, e, f, g, h, m, v = -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'ABV':
        if m_set_logic:
            print('(set-logic ABV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'BV':
        if m_set_logic:
            print('(set-logic BV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'AUFBV':
        if m_set_logic:
            print('(set-logic AUFBV)')
        set_options()
        ni, e, f, g, h, m, v, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'NIA':
        if m_set_logic:
            print('(set-logic NIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'ANIA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic ANIA)')
        else:
            if m_set_logic:
                print('(set-logic AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'AUFNIA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic AUFNIA)')
        else:
            if m_set_logic:
                print('(set-logic AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'AUFLIA':
        if m_set_logic:
            print('(set-logic AUFLIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'ALIA':
        if m_set_logic:
            print('(set-logic ALIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'LIA':
        if m_set_logic:
            print('(set-logic LIA)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'NRA':
        if m_set_logic:
            print('(set-logic NRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'LRA':
        if m_set_logic:
            print('(set-logic LRA)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'ANRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic ANRA)')
        else:
            if m_set_logic:
                print('(set-logic AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'AUFNRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic AUFNRA)')
        else:
            if m_set_logic:
                print('(set-logic AUFNIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    # lira_logics = ['QF_LIRA', 'LIRA', 'QF_ALIRA', 'ALIRA', 'QF_UFLIRA', 'UFLIRA', 'QF_AUFLIRA', 'AUFLIRA']
    # nira_logics = ['QF_NIRA', 'NIRA', 'QF_ANIRA', 'ANIRA', 'QF_UFNIRA',
    elif 'IRA' in logic_choice:
        if m_test_cvc4 or m_test_smtinterpol or m_test_yices:
            if m_set_logic:
                print('(set-logic ' + logic_choice + ')')
        # if m_set_logic: print('(set-logic ' + logic_choice + ')')
        elif m_test_diff:
            if m_set_logic:
                print('(set-logic ALL)')
        set_options()
        if 'QF' not in logic_choice:
            add_quantifiers = m_quantifier_rate
        add_reals = 1
        add_ints = 1
        v = m_create_exp_rate

        if logic_choice in ['QF_LIRA', 'QF_NIRA', 'LIRA', 'NIRA', 'QF_SLIRA']:
            a, b, c, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1

        elif logic_choice in ['QF_ALIRA', 'QF_ANIRA', 'ALIRA', 'ANIRA']:
            a, b, c, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, m_create_exp_rate

        elif logic_choice in ['QF_UFLIRA', 'QF_UFNIRA', 'UFLIRA', 'UFNIRA']:
            r, t, u, gen_arr = -1, -1, -1, -1

        elif logic_choice in ['QF_AUFLIRA', 'QF_AUFNIRA', 'AUFLIRA', 'AUFNIRA']:
            r, t, u, gen_arr = -1, -1, -1, 0.33

    elif logic_choice == 'AUFLRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic AUFLRA)')
        else:
            if m_set_logic:
                print('(set-logic AUFLIRA)')  # a trick for z3 to recognize
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'ALRA':
        if m_test_cvc4:
            if m_set_logic:
                print('(set-logic ALRA)')
        else:
            if m_set_logic:
                print('(set-logic AUFLIRA)')  # a trick for z3 to recognize
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = -1, -1, - \
            1, -1, -1, -1, -1, -1, -1, -1, m_create_exp_rate
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'RDL':
        if m_set_logic:
            print('(set-logic RDL)')
        set_options()
        a, b, c, ni, e, f, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UF':
        if m_set_logic:
            print('(set-logic UF)')
        set_options()
        ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UFBV':
        if m_set_logic:
            print('(set-logic UFBV)')
        set_options()
        ni, e, f, g, h, m, v, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UFNRA':
        if m_set_logic:
            print('(set-logic UFNRA)')
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UFLRA':
        if m_set_logic:
            print('(set-logic UFLRA)')
        set_options()
        ni, e, f, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_reals = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UFNIA':
        if m_set_logic:
            print('(set-logic UFNIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'UFLIA':
        if m_set_logic:
            print('(set-logic UFLIA)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'AX':
        if m_set_logic:
            print('(set-logic AX)')
        set_options()
        add_reals = 1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_BOOL' or logic_choice == 'BOOL':  # pure SAT
        if m_set_logic:
            print('(set-logic QF_UF)')  # should we?
        # else: if m_set_logic: print('(set-logic BV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    elif logic_choice == 'QBF':  # QBF
        if m_set_logic:
            print('(set-logic UF)')  # should we?
        # else: if m_set_logic: print('(set-logic BV)')
        set_options()
        a, b, c, ni, e, f, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'SET':
        # if m_test_cvc4: if m_set_logic: print('(set-logic QF_UFLIRAFS)') # shoud we?
        # if m_set_logic: print('(set-logic ALL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        # NOTE: BAPA does not support model generation?
        if random.random() < 0.3:
            add_reals = 1
            g, h, m, v, = m_create_exp_rate, m_create_exp_rate, m_create_exp_rate, m_create_exp_rate
        if random.random() < 0.5:
            add_quantifiers = m_quantifier_rate

    elif logic_choice == 'STRSET':
        # if m_test_cvc4: if m_set_logic: print('(set-logic QF_UFLIRAFS)') # shoud we?
        # if m_set_logic: print('(set-logic ALL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        if random.random() < 0.3:
            add_reals = 1
            g, h, m, v, = m_create_exp_rate, m_create_exp_rate, m_create_exp_rate, m_create_exp_rate

    # TODO: make QF_S and QF_SLIA different
    elif logic_choice == 'QF_S':
        # if m_test_cvc4: if m_set_logic: print('(set-logic QF_SLIA)') # shoud we?
        # if m_set_logic: print('(set-logic QF_S)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1  #
        # if random.random() < 0.5: add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QSTR':
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_SLIA':
        # if random.random() < 0.5: if m_set_logic: print('(set-logic QF_SLIA)')
        # else: if m_set_logic: print('(set-logic ALL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1

    elif logic_choice == 'SEQ':
        # if random.random() < 0.5:
        #    if m_set_logic: print('(set-logic QF_SLIA)')
        # else:
        #    if m_set_logic: print('(set-logic ALL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        # if random.random() < 0.5: add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_UFSLIA':
        # if random.random() < 0.5: if m_set_logic: print('(set-logic QF_UFSLIA)')
        # else: if m_set_logic: print('(set-logic ALL)')
        set_options()
        g, h, m, v, r, t, u, gen_arr = -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        # if random.random() < 0.5: add_quantifiers = m_quantifier_rate

    elif logic_choice == 'QF_SNIA':
        # if m_set_logic: print('(set-logic ALL)')
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        # if random.random() < 0.5: add_quantifiers = m_quantifier_rate

    elif logic_choice == 'SEPLOG':
        # Separation logic: CVC4 only
        set_options()
        a, b, c, g, h, m, v, r, t, u, gen_arr = - \
            1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        add_ints = 1
        # add_quantifiers = m_quantifier_rate

    return a, b, c, ni, e, f, g, h, m, v, r, t, u, gen_arr, add_ints, add_reals, add_quantifiers


class Clauses:
    def __init__(self, b, nc):
        int_nc = int(nc)
        self.n_clauses = int_nc
        self.clauses = []
        self.unused_options = list(b)
        self.all_options = list(b)

    def new_cnfs(self):
        global m_assert_id, m_all_assertions
        for i in range(self.n_clauses):
            cnf = "(or "
            cls_size = 0
            for j in range(2):
                n_left = ((self.n_clauses - i) * 3) + (3 - j)
                if len(self.unused_options) == n_left:
                    addition = random.choice(self.unused_options)
                    cnf += (str(addition) + " ")
                    cls_size += 1
                    self.unused_options.remove(addition)
                else:
                    addition = random.choice(self.all_options)
                    cnf += (str(addition) + " ")
                    cls_size += 1
                    if addition in self.unused_options:
                        self.unused_options.remove(addition)
            n_left = ((self.n_clauses - i) * 3) + (3 - j)
            if len(self.unused_options) == n_left:
                addition = random.choice(self.unused_options)
                cnf += (str(addition) + ")")
                cls_size += 1
            else:
                addition = random.choice(self.all_options)
                cnf += (str(addition) + ")")
                cls_size += 1
            self.clauses.append(cnf)

            if m_test_max_smt or m_test_max_sat:
                if random.random() < m_max_smt_rate:
                    print('(assert ' + cnf + ')')
                else:
                    # (assert -soft (= 0 0) :weight 1)
                    # print('(assert-soft ' + cnf + ')')
                    print('(assert-soft ' + cnf + ' :weight ' +
                          str(random.randint(1, 20)) + ' )')
            elif m_test_unsat_core or m_test_interpolant or m_test_proof or m_test_named_assert:
                m_assert_id += 1
                if m_test_smtinterpol or m_test_boolector or m_test_diff:
                    print(
                        '(assert (! (or ' + cnf + ' false) :named IP_' + str(m_assert_id) + '))')
                else:
                    print(
                        '(assert (! ' + cnf + ' :named IP_' + str(m_assert_id) + '))')
                m_all_assertions.append('IP_' + str(m_assert_id))

            else:
                if m_test_smtinterpol or m_test_boolector or m_test_diff:
                    print('(assert (or ' + cnf + ' false))')
                else:
                    print('(assert ' + cnf + ')')

    def new_dist_cnfs(self):
        global m_assert_id, m_all_assertions
        n_slots = (self.n_clauses * 3)
        tmp_string = ""
        for ith in range(n_slots - 1):
            n_left = n_slots - ith
            if len(self.unused_options) == n_left:
                addition = random.choice(self.unused_options)
                tmp_string += (str(addition) + "$")
                self.unused_options.remove(addition)
            else:
                addition = random.choice(self.all_options)
                tmp_string += (str(addition) + "$")
                if addition in self.unused_options:
                    self.unused_options.remove(addition)
        if len(self.unused_options) == 1:
            addition = random.choice(self.unused_options)
            tmp_string += str(addition)
        else:
            addition = random.choice(self.all_options)
            tmp_string += str(addition)

        place_holders = find(tmp_string, '$')
        w = n_slots - (self.n_clauses - 1)
        spaces = random.sample(place_holders, w)
        for x in spaces:
            tmp_string = replace_idx(tmp_string, x, ' ')
        partitions = find(tmp_string, '$')
        CNFs = []
        for x in partitions:
            c = tmp_string[:x]
            q = c.rfind('$')
            if q >= 0:
                c = c[q + 1:]
            CNFs.append(c)
        for items in CNFs:
            new_CNF = '(or {})'.format(items)
            if m_test_smtinterpol or m_test_boolector or m_test_diff:
                new_CNF = '(or {} false)'.format(
                    items)  # corrent?
            # print("; new_dist_cnfs")
            self.clauses.append(new_CNF)
            if m_test_max_smt or m_test_max_sat:
                if random.random() < m_max_smt_rate:
                    print('(assert {})'.format(new_CNF))
                else:
                    print('(assert-soft {} :weight {})'.format(new_CNF,
                                                               str(random.randint(1, 20))))
            elif m_test_unsat_core or m_test_interpolant or m_test_proof or m_test_named_assert:
                m_assert_id += 1
                print(
                    '(assert (! ' +
                    format(new_CNF) +
                    ' :named IP_' +
                    str(m_assert_id) +
                    '))')
                m_all_assertions.append('IP_' + str(m_assert_id))
            else:
                print('(assert {})'.format(new_CNF))

    def cnf_choice(self):
        return random.choice(self.clauses)

    def node_from_cnf(self):
        n_operands = random.randint(1, 10)
        operands = str(random.choice(self.clauses))
        for _ in range(n_operands):
            operands += (" " + str(random.choice(self.clauses)))
        # TODO; seems bugs here: too many 'or' in the smt2 files
        # thus, I add a number m_or_and_rate
        if random.random() < 0.5:
            n_and = operands.count('and')
            n_or = operands.count('or')
            if n_and > n_or:
                new_cnf = Op('or', operands)
            elif n_and < n_or:
                new_cnf = Op('and', operands)
            else:
                if random.random() < 0.5:
                    new_cnf = Op('or', operands)
                else:
                    new_cnf = Op('and', operands)
        else:
            # if random.random() < 1: new_cnf = Op('or', operands)
            if random.random() < m_or_and_rate:
                new_cnf = Op('or', operands)
            else:
                new_cnf = Op('and', operands)

        self.clauses.append(new_cnf)
        return new_cnf

    def bin_node(self):
        op1 = '{} {}'.format(
            random.choice(
                self.clauses), random.choice(
                self.clauses))
        op2 = '{} {}'.format(
            random.choice(
                self.clauses), random.choice(
                self.clauses))
        new_cnf1 = Op('=>', op1)
        new_cnf2 = Op('or', op2)
        self.clauses.append(new_cnf1)
        self.clauses.append(new_cnf2)
        return new_cnf1, new_cnf2


class SimpleNodes:
    def __init__(self, init_vars, ty):
        self.d = OrderedDict()
        self.d[Bool()] = []
        self.d[Int()] = []
        self.d[Real()] = []
        self.d[String()] = []
        self.dict = OrderedDict()
        self.dict[Bool()] = 0
        self.dict[Int()] = 0
        self.dict[Real()] = 0
        self.dict[String()] = 0
        self.new_keys = []
        self.indices = []

        if m_test_datalog_chc:
            global IntBinOp
            global IntNOp
            global RealBinOp
            global RealNOp
            IntBinOp = ["-", "+"]
            IntNOp = ["-", "+"]
            RealBinOp = ["-", "+"]
            RealNOp = ["-", "+"]

        if ty == "Int":
            for variable in init_vars:
                self.d[Int()].append(variable)
            self.d[Int()].append(str(random.randint(0, 5000)))
            self.d[Int()].append(str(random.randint(0, 5000)))
            for _ in range(15):
                self.int_from_int()
                self.bool_from_int()
        elif ty == "Real":
            for variable in init_vars:
                self.d[Real()].append(variable)
            self.d[Real()].append(str(random.randint(0, 5000)))
            self.d[Real()].append(str(random.randint(0, 5000)))
            for _ in range(15):
                self.real_from_real()
                self.bool_from_real()
        elif ty == "String":
            for variable in init_vars:
                self.d[String()].append(variable)
            for _ in range(15):
                self.string_from_string()

    def get_int_term(self):
        return random.choice(self.d[Int()])

    def get_real_term(self):
        return random.choice(self.d[Real()])

    def get_string_term(self):
        return random.choice(self.d[String()])

    def get_bool(self):
        return random.choice(self.d[Bool()])

    def string_from_string(self):
        chance = random.random()
        if chance < 0.1:  # unary
            new_str = "\"" + random_string() + "\""
            self.d[String()].append(new_str)
        elif chance < 0.45:  # binary
            par = random.choice(self.d[String()])
            operands = str(par)
            par = random.choice(self.d[String()])
            operands += (" " + str(par))
            new_str = String_Op(random.choice(StringBinOp), operands)
            self.d[String()].append(new_str)
        else:
            par = random.choice(self.d[String()])
            operands = str(par)
            n = random.randrange(1, 5)
            for _ in range(n):
                if random.random() < 0.7:
                    par = random.choice(self.d[String()])
                    operands += (" " + str(par))
                else:
                    substr = random_string()
                    operands += (" " + "\"" + substr + "\"")
            op_to_use = random.choice(StringNOp)
            new_str = String_Op(op_to_use, operands)
            self.d[String()].append(new_str)

    def bool_from_string(self):
        chance = random.random()
        if chance < 0.5:
            par = random.choice(self.d[String()])
            operands = str(par)
            par = random.choice(self.d[String()])
            operands += (" " + str(par))
            new_bool = Bool_Op(random.choice(StringBinBoolOp), operands)
            self.d[Bool()].append(new_bool)
        else:
            par = random.choice(self.d[String()])
            operands = str(par)
            op_to_use = random.choice(StringNBoolOp)
            n = random.randrange(1, 5)
            for _ in range(n):
                if random.random() < 0.9:
                    par = random.choice(self.d[String()])
                    operands += (" " + str(par))
                else:
                    if op_to_use == "distinct":
                        substr = random_string()
                        operands += (" " + "\"" + substr + "\"")
                    else:
                        par = random.choice(self.d[String()])
                        operands += (" " + str(par))
            new_bool = Bool_Op(op_to_use, operands)
            self.d[Bool()].append(new_bool)

    def int_from_int(self):
        p = random.random()
        if p < 0.3:
            par = random.choice(self.d[Int()])
            new_int = Int_Op(random.choice(IntUnOp), par)
            self.d[Int()].append(new_int)
        elif p < 0.66:
            par = random.choice(self.d[Int()])
            operand = str(par)
            par2 = random.choice(self.d[Int()])
            operand += (" " + str(par2))
            new_int = Int_Op(random.choice(IntBinOp), operand)
            self.d[Int()].append(new_int)
        else:
            par = random.choice(self.d[Int()])
            operand = str(par)
            n = random.randrange(1, 5)
            for _ in range(n):
                par = random.choice(self.d[Int()])
                operand += (" " + str(par))
            op_to_use = random.choice(IntNOp)
            new_int = Int_Op(op_to_use, operand)
            self.d[Int()].append(new_int)

    def bool_from_int(self):
        if random.random() < 0.66:  # seems the old strategy is "better"?
            par = random.choice(self.d[Int()])
            operand = str(par)
            par = random.choice(self.d[Int()])
            operand += (" " + str(par))
            new_bool = Bool_Op(random.choice(IRNBoolOp), operand)
            self.d[Bool()].append(new_bool)
            return  # stop here
        par = random.choice(self.d[Int()])
        operands = str(par)
        n_operands = random.randrange(1, 6)
        for _ in range(n_operands):
            par = random.choice(self.d[Int()])
            operands += (" " + str(par))
        new_bool = Bool_Op(random.choice(IRNBoolOp), operands)
        self.d[Bool()].append(new_bool)

    def real_from_real(self):
        p = random.random()
        if p < 0.3:
            par = random.choice(self.d[Real()])
            new_r = Real_Op(random.choice(RealUnOp), par)
            self.d[Real()].append(new_r)
        elif p < 0.66:
            par = random.choice(self.d[Real()])
            operand = str(par)
            par2 = random.choice(self.d[Real()])
            operand += (" " + str(par2))
            new_r = Real_Op(random.choice(RealBinOp), operand)
            self.d[Real()].append(new_r)
        else:
            par = random.choice(self.d[Real()])
            operand = str(par)
            n = random.randrange(1, 5)
            for _ in range(n):
                par = random.choice(self.d[Real()])
                operand += (" " + str(par))
            new_r = Real_Op(random.choice(RealNOp), operand)
            self.d[Real()].append(new_r)

    def bool_from_real(self):
        par = random.choice(self.d[Real()])
        operands = str(par)
        n_operands = random.randrange(1, 6)
        for _ in range(n_operands):
            par = random.choice(self.d[Real()])
            operands += (" " + str(par))
        new_bool = Bool_Op(random.choice(IRNBoolOp), operands)
        self.d[Bool()].append(new_bool)

    def bool_from_fp(self, fp_sort=None):
        # If no specific FP sort is provided, use a random existing one
        if fp_sort is None:
            # Get all FP sorts currently in use
            fp_sorts = [s for s in self.d.keys() if isinstance(s, FP)]
            if not fp_sorts:
                # If no FP sorts exist yet, create a default one
                if m_test_fp64:
                    fp_sort = FP(11, 53)  # double precision
                else:
                    fp_sort = FP(8, 24)   # single precision
            else:
                # Select a random existing FP sort
                fp_sort = random.choice(fp_sorts)
                
        # Get or create FP values of this sort
        if fp_sort not in self.d or not self.d[fp_sort]:
            self.new_fp(fp_sort.eb, fp_sort.sb)
            
        # n-array or binary?
        if random.random() < 0.15:
            # Unary operation
            par = random.choice(self.d[fp_sort])
            operand = str(par)
            new_bool = Bool_Op(random.choice(UnFPBoolOp), operand)
            self.d[Bool()].append(new_bool)
            self.dict[Bool()] += 1
            return
        
        # Create a boolean comparison between FP values
        par = random.choice(self.d[fp_sort])
        operands = str(par)
        n_operands = random.randrange(1, 5)
        for _ in range(n_operands):
            par = random.choice(self.d[fp_sort])
            operands += (" " + str(par))
        new_bool = Bool_Op(random.choice(IRNFPBoolOp), operands)
        self.d[Bool()].append(new_bool)
        # give possibility of asserting this new bool here?
        self.dict[Bool()] += 1


class Nodes:
    def __init__(self, a_ints, a_reals):
        self.d = OrderedDict()
        self.d[Bool()] = []

        self.nq = 0
        self.qdict = OrderedDict()

        # dictionary of number of all nodes ever created
        self.dict = OrderedDict()
        self.dict[Bool()] = 0
        self.dict[Int()] = 0
        self.dict[Real()] = 0

        self.initial_ints = a_ints
        self.initial_reals = a_reals

        self.new_keys = []
        self.indices = []

        self.n_vars = random.randint(1, 6)
        self.n_ints = random.randint(1, 20)
        self.n_reals = random.randint(1, 20)

        self.initial_bvs = 0  # bv
        self.n_bvs = 0

        # just for storing quantified exp? (TODO: not so easy, becaues we need to manager it at push/pop
        # self.dict[Quantifier()] = 1
        # self.d[Quantifier()] = []

        if (m_test_fp or m_test_fp_lra) and (not m_test_datalog_chc):
            self.dict[FP()] = 0
            self.initial_fps = 1
            self.n_fps = random.randint(1, 20)
            self.d[FP()] = []

            self.initial_ints = 0

            if not m_test_fp_lra:
                self.initial_reals = 0

            for ith in range(self.n_fps):
                if not m_test_fp_no_num and random.random() < 0.4:
                    new_fp = random_fp()
                    self.d[FP()].append(new_fp)
                    self.dict[FP()] += 1
                else:
                    self.d[FP()].append(Var_FP(ith))
                    if m_test_fp64:
                        print('(declare-fun {} () Float64)'.format(Var_FP(ith)))
                    else:
                        print('(declare-fun {} () Float32)'.format(Var_FP(ith)))
                    self.dict[FP()] += 1
            if m_test_fp64:  # special FP constants
                self.d[FP()].append("(_ +oo 11 53)")
                self.d[FP()].append("(_ -oo 11 53)")
                self.d[FP()].append("(_ +zero 11 53)")
                self.d[FP()].append("(_ -zero 11 53)")
                self.d[FP()].append("(_ NaN 11 53)")
            else:
                self.d[FP()].append("(_ +oo 8 24)")
                self.d[FP()].append("(_ -oo 8 24)")
                self.d[FP()].append("(_ +zero 8 24)")
                self.d[FP()].append("(_ +zero 8 24)")
                self.d[FP()].append("(_ NaN 8 24)")

        if m_test_datalog_chc:
            self.int_funcs = []
            self.real_funcs = []
            self.bv_funcs = []

            # (declare-fun inv-int1 (Real Real ) Bool)
            self.n_vars = 3
            if m_test_datalog_chc_logic == "int":
                self.initial_ints = 1
                self.initial_reals = 0
                self.initial_bvs = 0
                self.n_ints = 5
            elif m_test_datalog_chc_logic == "real":
                self.initial_reals = 1
                self.initial_ints = 0
                self.initial_bvs = 0
                self.n_reals = 5
            elif m_test_datalog_chc_logic == "bv":
                self.initial_bvs = 1
                self.initial_ints = 0
                self.initial_reals = 0
                self.n_bvs = 8

            '''
            if self.initial_ints > 0:
                self.int_funcs.append("inv-int1")
                self.int_funcs.append("inv-int2")
                self.int_funcs.append("inv-int3")
                print("(declare-fun inv-int1 (Int Int) Bool)")
                print("(declare-fun inv-int2 (Int Int Int) Bool)")
                print("(declare-fun inv-int3 (Int Int Int Int) Bool)")

            elif self.initial_reals > 0:
                self.real_funcs.append("inv-real1")
                self.real_funcs.append("inv-real2")
                self.real_funcs.append("inv-real3")
                print("(declare-fun inv-real1 (Real Real) Bool)")
                print("(declare-fun inv-real2 (Real Real Real) Bool)")
                print("(declare-fun inv-real3 (Real Real Real Real) Bool)")
            '''
            return

        if m_test_my_uf:
            print("(declare-fun uf3 (Bool Bool Bool) Bool)")
            print("(declare-fun uf4 (Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf5 (Bool Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf6 (Bool Bool Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf7 (Bool Bool Bool Bool Bool Bool Bool) Bool)")

            print("(declare-fun uf3_2 (Bool Bool Bool) Bool)")
            print("(declare-fun uf4_2 (Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf5_2 (Bool Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf6_2 (Bool Bool Bool Bool Bool Bool) Bool)")
            print("(declare-fun uf7_2 (Bool Bool Bool Bool Bool Bool Bool) Bool)")

        for ith in range(self.n_vars):
            self.d[Bool()].append(Var_Bool(ith))
            print('(declare-fun {} () Bool)'.format(Var_Bool(ith)))
            self.dict[Bool()] += 1

        if self.initial_ints == 1:
            self.d[Int()] = []
            for ith in range(self.n_ints):
                if random.random() < 0.5 and (not m_test_string):  # donot show Int for QF_S
                    self.d[Int()].append(Var_Int(ith))
                    print('(declare-fun {} () Int)'.format(Var_Int(ith)))
                else:
                    val = random.randint(0, 100)
                    self.d[Int()].append(val)
                self.dict[Int()] += 1

            if m_test_seplog:
                print("(declare-heap (Int Int))")
            if m_test_my_uf:
                print("(declare-fun ufib3 (Int Int Int) Bool)")
                print("(declare-fun ufib4 (Int Int Int Int) Bool)")
                print("(declare-fun ufib5 (Int Int Int Int Int) Bool)")
                print("(declare-fun ufib6 (Int Int Int Int Int Int) Bool)")
                print("(declare-fun ufib7 (Int Int Int Int Int Int Int) Bool)")

                print("(declare-fun ufbi3 (Bool Bool Bool) Int)")
                print("(declare-fun ufbi4 (Bool Bool Bool Bool) Int)")
                print("(declare-fun ufbi5 (Bool Bool Bool Bool Bool) Int)")
                print("(declare-fun ufbi6 (Bool Bool Bool Bool Bool Bool) Int)")
                print("(declare-fun ufbi7 (Bool Bool Bool Bool Bool Bool Bool) Int)")

                print("(declare-fun ufii3 (Int Int Int) Int)")
                print("(declare-fun ufii4 (Int Int Int Int) Int)")
                print("(declare-fun ufii5 (Int Int Int Int Int) Int)")
                print("(declare-fun ufii6 (Int Int Int Int Int Int) Int)")
                print("(declare-fun ufii7 (Int Int Int Int Int Int Int) Int)")

                print("(declare-fun ufii3_2 (Int Int Int) Int)")
                print("(declare-fun ufii4_2 (Int Int Int Int) Int)")
                print("(declare-fun ufii5_2 (Int Int Int Int Int) Int)")
                print("(declare-fun ufii6_2 (Int Int Int Int Int Int) Int)")
                print("(declare-fun ufii7_2 (Int Int Int Int Int Int Int) Int)")

        if self.initial_reals == 1:
            self.d[Real()] = []
            for ith in range(self.n_reals):
                if random.random() < 0.5:
                    self.d[Real()].append(Var_Real(ith))
                    print('(declare-fun {} () Real)'.format(Var_Real(ith)))
                else:
                    new_real = random_real()
                    self.d[Real()].append(new_real)
                self.dict[Real()] += 1
            if m_test_my_uf:
                print("(declare-fun ufrb3 (Real Real Real) Bool)")
                print("(declare-fun ufrb4 (Real Real Real Real) Bool)")
                print("(declare-fun ufrb5 (Real Real Real Real Real) Bool)")
                print("(declare-fun ufrb6 (Real Real Real Real Real Real) Bool)")
                print("(declare-fun ufrb7 (Real Real Real Real Real Real Real) Bool)")

                print("(declare-fun ufbr3 (Bool Bool Bool) Real)")
                print("(declare-fun ufbr4 (Bool Bool Bool Bool) Real)")
                print("(declare-fun ufbr5 (Bool Bool Bool Bool Bool) Real)")
                print("(declare-fun ufbr6 (Bool Bool Bool Bool Bool Bool) Real)")
                print("(declare-fun ufbr7 (Bool Bool Bool Bool Bool Bool Bool) Real)")

                print("(declare-fun ufrr3 (Real Real Real) Real)")
                print("(declare-fun ufrr4 (Real Real Real Real) Real)")
                print("(declare-fun ufrr5 (Real Real Real Real Real) Real)")
                print("(declare-fun ufrr6 (Real Real Real Real Real Real) Real)")
                print("(declare-fun ufrr7 (Real Real Real Real Real Real Real) Real)")

        if self.initial_bvs == 1:  # seems not used
            for ith in range(self.n_bvs):
                if random.random() < 0.25:
                    width = random.randint(1, 64)
                    # width = random.randint(10000, 100000)
                    bv_sort = BV(width)
                    if bv_sort not in self.d.keys():
                        self.d[bv_sort] = []
                        self.dict[bv_sort] = 0
                    const = Var_BV(width, len(self.d[bv_sort]))
                    print('(declare-fun {} () {})'.format(const, bv_sort))
                    self.d[bv_sort].append(const)
                    self.dict[bv_sort] += 1
                else:
                    bv, width = random_BV()
                    bv_sort = BV(width)
                    if bv_sort not in self.d.keys():
                        self.d[bv_sort] = []
                        self.dict[bv_sort] = 0
                        self.d[bv_sort].append(bv)
                        self.dict[bv_sort] += 1

        if m_test_set_bapa:
            self.dict[Set()] = 0
            self.d[Set()] = []
            for ith in range(15):
                self.d[Set()].append(Var_Set(ith))
                if m_test_str_set_bapa:
                    print('(declare-fun {} () (Set String))'.format(Var_Set(ith)))
                else:
                    print('(declare-fun {} () (Set Int))'.format(Var_Set(ith)))
                self.dict[Set()] += 1
            if m_test_set_eq and (not m_test_str_set_bapa):
                # if True:
                print("(declare-fun seteq ((Set Int) (Set Int)) Bool)")
                print(
                    "(assert (forall ((?s1 (Set Int)) (?s2 (Set Int))) (= (seteq ?s1 ?s2) (= ?s1 ?s2))))")
                print(
                    "(assert (forall ((?s1 (Set Int)) (?s2 (Set Int))) (= (seteq ?s1 ?s2) (and (subset ?s1 ?s2) (subset ?s2 ?s1)))))")

        if m_test_bag_bapa:
            self.dict[Bag()] = 0
            self.d[Bag()] = []
            for ith in range(15):
                self.d[Bag()].append(Var_Bag(ith))
                print('(declare-fun {} () (Bag Int))'.format(Var_Bag(ith)))
                self.dict[Bag()] += 1

        if (m_test_string or m_test_string_lia) and (not m_test_datalog_chc):
            self.dict[String()] = 0
            self.d[String()] = []
            self.dict[Regular()] = 0
            self.d[Regular()] = []
            self.dict[Seq()] = 0
            self.d[Seq()] = []
            nstr = random.randint(5, 20)
            if m_test_seq:
                for ith in range(nstr):
                    self.d[Seq()].append(Var_Seq(ith))
                    print('(declare-fun {} () (Seq Int))'.format(Var_Seq(ith)))
                    self.dict[Seq()] += 1
                # return #TEMP: mixing seq and str

            for ith in range(nstr):
                self.d[String()].append(Var_String(ith))
                print('(declare-fun {} () String)'.format(Var_String(ith)))
                self.dict[String()] += 1

            if m_test_my_uf:
                print("(declare-fun ufss3 (String String String) String)")
                print("(declare-fun ufss4 (String String String String) String)")
                print("(declare-fun ufss5 (String String String String String) String)")
                print(
                    "(declare-fun ufss6 (String String String String String String) String)")
                print(
                    "(declare-fun ufss7 (String String String String String String String) String)")

                print("(declare-fun ufss3_2 (String String String) String)")
                print("(declare-fun ufss4_2 (String String String String) String)")
                print(
                    "(declare-fun ufss5_2 (String String String String String) String)")
                print(
                    "(declare-fun ufss6_2 (String String String String String String) String)")
                print(
                    "(declare-fun ufss7_2 (String String String String String String String) String)")

            # init someRE?
            # '''
            if m_test_string_re:
                for ith in range(15):
                    if random.random() < 0.65:
                        par = random.choice(self.d[String()])
                    else:
                        par = "\"" + random_string(10) + "\""
                    operands = str(par)
                    new_re = Regular_Op('str.to_re', operands)
                    self.d[Regular()].append(new_re)

                    self.d[Regular()].append("re.allchar")
                    self.d[Regular()].append("re.all")
                    self.d[Regular()].append("re.none")

        if m_test_recfun:
            self.define_rec_fun()

    # TODO: maintain the "liveness" of named assertions when using push/pop

    def push(self, k=1):
        global m_all_assertions, m_backtrack_points

        print('(push ' + str(k) + ')')

        for ss in range(k):
            self.new_keys.append(len(list(self.d)))

            self.indices.append([])
            for key in self.d:
                self.indices[-1].append(len(self.d[key]))

            point = len(m_all_assertions)
            m_backtrack_points.append(point)

    def pop(self, k=1):
        global m_all_assertions, m_backtrack_points

        print('(pop ' + str(k) + ')')

        for ss in range(k):
            n_keys = self.new_keys[-1]
            self.new_keys.pop()
            added_keys = list(self.d)[n_keys:]
            for ones in added_keys:
                del self.d[ones]

            for key in self.d:
                jj = self.indices[-1][list(self.d).index(key)]
                del self.d[key][jj:]
            self.indices.pop()

            point = m_backtrack_points.pop()
            m_all_assertions = m_all_assertions[0:point]

    #
    def define_rec_fun(self):
        # return
        # Define recustive function
        # I will reuse qdict
        simp = SimpleNodes(['x', 'y'], "Int")
        term = simp.get_int_term()
        print("(define-fun recfun2 ((x Int) (y Int)) Int " + str(term) + ")")

        simp = SimpleNodes(['x', 'y', 'z'], "Int")
        term = simp.get_int_term()
        print("(define-fun recfun3 ((x Int) (y Int) (z Int)) Int " + str(term) + ")")

    # only for CHC
    def quantifier_chc(self):
        # TODO: choose the number of variales in each rule!!!!!!!
        # Not just 3
        # try m_test_datalog_chc_var_bound

        sorted_var = '('
        n = random.randint(0, m_test_datalog_chc_var_bound)
        for _ in range(n):
            ovar = Var_Quant(self.nq)
            self.nq += 1
            # osort = random.choice(list(self.d))
            osort = None
            if m_test_datalog_chc_logic == "int":
                for o in list(self.d):
                    if isinstance(o, Int):
                        osort = o
                        break
            elif m_test_datalog_chc_logic == "real":
                for o in list(self.d):
                    if isinstance(o, Real):
                        osort = o
                        break
            elif m_test_datalog_chc_logic == "bv":
                for o in list(self.d):
                    if isinstance(o, BV):
                        osort = o
                        break

            if osort and (osort not in self.qdict):
                self.qdict[osort] = []
            self.qdict[osort].append(ovar)
            osv = '({} {}) '.format(ovar, osort)
            sorted_var += osv

        ovar = Var_Quant(self.nq)
        self.nq += 1
        osort = random.choice(list(self.d))
        if osort not in self.qdict:
            self.qdict[osort] = []
        self.qdict[osort].append(ovar)
        osv = '({} {}))'.format(ovar, osort)
        sorted_var += osv

        # try more multile times?
        times_batch = 1  # how many assertions for a group of quantified vars.
        for _ in range(0, times_batch):
            stat, term = self.qterm_chc()
            # print("stat, term", stat, term)
            if stat:
                statement = '(assert (forall {} {}))'.format(sorted_var, term)
                print(statement)

        self.qdict.clear()

    def qterm_chc(self):
        qkeys = list(self.qdict)
        nsam = 2
        if len(qkeys) < 2:
            return False, "Error"
        qkeys = random.sample(qkeys, nsam)
        boolean_subexpressions = ""
        for ith in qkeys:
            subexpr = self.qsubexpression_chc(ith)
            boolean_subexpressions += (str(subexpr) + " ")
        boolean_subexpressions = boolean_subexpressions[:-1]
        if nsam == 2:  # binary
            term = '({} {})'.format(
                random.choice(BiOp),
                boolean_subexpressions)
            return True, term

    def quantifier(self):
        global m_assert_id, m_all_assertions
        sorted_var = '('
        var_list = []
        # n = random.randint(0, 3)
        n = random.randint(0, 7)
        if m_test_boolector:
            n = random.randint(0, 5)
        for _ in range(n):
            ovar = Var_Quant(self.nq)
            self.nq += 1
            osort = random.choice(list(self.d))
            if m_test_boolector and isinstance(osort, Arr):
                continue  # TODO: for boolector.
            if osort not in self.qdict:
                self.qdict[osort] = []
            self.qdict[osort].append(ovar)
            osv = '({} {}) '.format(ovar, osort)
            sorted_var += osv
            var_list.append(osv)
        if m_test_boolector and m_global_logic == 'ABV':
            sorted_var += '(q13145926 Bool))'
        else:
            ovar = Var_Quant(self.nq)
            self.nq += 1
            osort = random.choice(list(self.d))
            if osort not in self.qdict:
                self.qdict[osort] = []
            self.qdict[osort].append(ovar)
            osv = '({} {}))'.format(ovar, osort)
            sorted_var += osv
            var_list.append('({} {})'.format(ovar, osort))

        term = self.qterm()

        if m_test_cvc4 and m_test_qe:
            termone = self.qterm()
            termtwo = self.qterm()
            stmtx = ' (forall {} {})'.format(sorted_var, termtwo)
            stmty = ' (exists {} {})'.format(sorted_var, termtwo)
            if random.random() < 0.5:
                fstt = '(forall {}'.format(sorted_var)
            else:
                fstt = '(exists {}'.format(sorted_var)
            # fstt += ' (or {} '.format(termone)
            fstt += ' (' + random.choice(['or', 'xor',
                                          'and', 'or']) + ' {} '.format(termone)
            if random.random() < 0.5:
                fstt += stmtx
            else:
                fstt += stmty
            for _ in range(8):
                tmp = random.choice(self.d[Bool()])
                fstt += ' {} '.format(tmp)
            fstt += '))'
            # self.d[Bool()].append(fstt) # stack cannot have quantified
            # formulas?
            if random.random() < 0.5:
                print('(get-qe ' + fstt + ')')
            else:
                print('(get-qe-disjunct ' + fstt + ')')
            self.qdict.clear()
            return

        if random.random() < 0.5:
            notqterm = random.random()
            if notqterm < 0.5:
                if random.random() < 0.45:
                    statement = '(forall {} {})'.format(sorted_var, term)
                else:
                    statement = '(exists {} {})'.format(sorted_var, term)

                self.d[Bool()].append(statement)
            else:
                if random.random() < 0.45:
                    statement = '(not (forall {} {}))'.format(sorted_var, term)
                else:
                    statement = '(not (exists {} {}))'.format(sorted_var, term)

                self.d[Bool()].append(statement)

        if not m_use_fancy_qterm:  # or m_test_yices: # NO fancy qterm for Yices
            self.qdict.clear()
            return
        elif random.random() < 0.5:
            termone = self.qterm()
            termtwo = self.qterm()
            if random.random() < 0.25:  # experimetal: try forall exists randomly
                prex = ''
                for var in var_list:
                    if random.random() < 0.5:
                        prex = prex + ' (forall ({}) '.format(var)
                    else:
                        prex = prex + ' (exists ({}) '.format(var)

                fst = ' (' + random.choice(['or', 'xor',
                                            'and', 'or']) + ' {} '.format(termone)
                for _ in range(5):
                    tmp = random.choice(self.d[Bool()])
                    fst += ' {} '.format(tmp)

                new_bool = prex + fst + ')'
                for _ in var_list:
                    new_bool = new_bool + ')'

                self.d[Bool()].append(new_bool)
                self.qdict.clear()  # why clear?
                return

            # the orginal fancy
            stmtx = ' (forall {} {})'.format(sorted_var, termtwo)
            stmty = ' (exists {} {})'.format(sorted_var, termtwo)

            if random.random() < 0.5:
                fstt = '(forall {}'.format(sorted_var)
            else:
                fstt = '(exists {}'.format(sorted_var)

            # fstt += ' (or {} '.format(termone)  # should we?
            fstt += ' (' + random.choice(['or', 'xor',
                                          'and', 'or']) + ' {} '.format(termone)
            if random.random() < 0.5:
                fstt += stmtx
            else:
                fstt += stmty
            for _ in range(5):
                tmp = random.choice(self.d[Bool()])
                fstt += ' {} '.format(tmp)
            fstt += '))'
            new_bool = fstt
            if random.random() < 0.5:
                new_bool = '(not ' + fstt + ')'
            self.d[Bool()].append(new_bool)
            # print('(assert ' + fstt + ' )')

        self.qdict.clear()  # why clear?

    def qterm(self):
        qkeys = list(self.qdict)
        nsam = random.randint(0, len(self.qdict.keys()))
        qkeys = random.sample(qkeys, nsam)
        if nsam == 0:
            term = random.choice(self.d[Bool()])
        boolean_subexpressions = ""
        for ith in qkeys:
            subexpr = self.qsubexpression(ith)
            boolean_subexpressions += (str(subexpr) + " ")
        boolean_subexpressions = boolean_subexpressions[:-1]
        if nsam == 1:  # unary
            term = '({} {})'.format(
                random.choice(UnOp),
                boolean_subexpressions)
        elif nsam == 2:  # binary
            term = '({} {})'.format(
                random.choice(BiOp),
                boolean_subexpressions)
        elif nsam > 2:  # n-array
            term = '({} {})'.format(
                random.choice(NarOp),
                boolean_subexpressions)

        return term

    def extend_qdict_array(self):
        # TOOD: too complex??...
        return None

    def extend_qdict_bv(self):
        options = []
        qkeys = list(self.qdict)
        for sort in qkeys:
            if isinstance(sort, BV):
                options.append(sort)

        if len(options) > 0:
            s = random.choice(options)
            prob = random.random()
            if prob < 0.05:  # concat
                s2 = random.choice(options)
                width = s.w + s2.w
                par1 = random.choice(self.qdict[s])
                par2 = random.choice(self.qdict[s2])
                operand = str(par1) + " " + str(par2)
                new_BV = BV_Op("concat", operand)
                bv_sort = BV(width)
                if bv_sort not in qkeys:
                    self.qdict[bv_sort] = []
                self.qdict[bv_sort].append(new_BV)

            elif prob < 0.1 and (not m_test_eldarica):  # repeat
                ii = random.randint(1, 10)
                width = ii * s.w
                operator = '(_ repeat {})'.format(ii)
                par = random.choice(self.qdict[s])
                new_BV = BV_Op(operator, par)
                bv_sort = BV(width)
                if bv_sort not in qkeys:
                    self.qdict[bv_sort] = []
                self.qdict[bv_sort].append(new_BV)

            elif prob < 0.2:  # unary, extract
                if random.random() < 0.5:  # unary
                    par = random.choice(self.qdict[s])
                    new_BV = BV_Op(random.choice(Un_BV_BV), par)
                    self.qdict[s].append(new_BV)
                else:  # extract
                    width = s.w
                    parameter1 = random.randrange(0, width)
                    parameter2 = random.randint(0, parameter1)
                    operator = "(_ extract {} {})".format(
                        parameter1, parameter2)
                    new_width = parameter1 - parameter2 + 1
                    par = random.choice(self.qdict[s])
                    new_BV = BV_Op(operator, par)
                    bv_sort = BV(new_width)
                    if bv_sort not in qkeys:
                        self.qdict[bv_sort] = []
                    self.qdict[bv_sort].append(new_BV)

            elif prob < 0.3:
                ii = random.randint(0, 10)
                if random.random() < 0.5:
                    if random.random() < 0.5:
                        operator = "(_ zero_extend {})".format(ii)
                    else:
                        operator = "(_ sign_extend {})".format(ii)
                    width = s.w + ii
                    par = random.choice(self.qdict[s])
                    new_BV = BV_Op(operator, par)
                    bv_sort = BV(width)
                    if bv_sort not in qkeys:
                        self.qdict[bv_sort] = []
                    self.qdict[bv_sort].append(new_BV)
                elif not m_test_eldarica:  # for Eldarica CHC
                    if random.random() < 0.5:
                        operator = "(_ rotate_left {})".format(ii)
                    else:
                        operator = "(_ rotate_right {})".format(ii)
                    par = random.choice(self.qdict[s])
                    new_BV = BV_Op(operator, par)
                    self.qdict[s].append(new_BV)

            elif prob < 0.4 and not m_test_eldarica:  # n-array
                a = random.randint(1, 3)
                par = random.choice(self.qdict[s])
                operand = str(par)
                for ii in range(a):
                    par = random.choice(self.qdict[s])
                    operand += (" " + str(par))
                new_BV = BV_Op(random.choice(N_BV_BV), operand)
                self.qdict[s].append(new_BV)

            else:  # binary
                par1 = random.choice(self.qdict[s])
                par2 = random.choice(self.qdict[s])
                operand = str(par1) + " " + str(par2)
                operator = random.choice(Bin_BV_BV)
                new_BV = BV_Op(operator, operand)
                if operator == "bvcomp":
                    if BV(1) not in qkeys:
                        self.qdict[BV(1)] = []
                    self.qdict[BV(1)].append(new_BV)
                else:
                    self.qdict[s].append(new_BV)
