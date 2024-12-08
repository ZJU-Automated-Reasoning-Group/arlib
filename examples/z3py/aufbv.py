"""
Test AUFBV?
"""

from z3 import *


# "qfufbv_ackr", "A tactic for solving QF_UFBV based on Ackermannization."
def aufbv2bv(exp):
    """
    FIXME:
    - sometimes, there may exist quantifier after bvarray2uf
    - sometimes, there still exist UF after ackermannize_bv
    """
    tac = AndThen("simplify", "bvarray2uf", "ackermannize_bv")
    # tac = AndThen("simplify", "bvarray2uf")
    # tac = AndThen("simplify", "bvarray2uf", "qfufbv_ackr")
    return simplify(tac.apply(exp).as_expr())


def to_smt2(exp):
    s = Solver()
    s.add(exp)
    return s.to_smt2()


def test():
    A = Array('A', BitVecSort(8), BitVecSort(8))
    x, y = BitVecs('x y', 8)
    exp = And(A[x] == y, x >= y)
    print(exp)
    tt = aufbv2bv(exp)
    print(to_smt2(tt))


def test2():
    fml_str = '''
(set-logic QF_AUFBV )
(declare-fun a () (Array (_ BitVec 32) (_ BitVec 8) ) )
(assert (let ( (?B1 (concat  (select  a (_ bv3 32) ) (concat  (select  a (_ bv2 32) ) (concat  (select  a (_ bv1 32) ) (select  a (_ bv0 32) ) ) ) ) ) ) (and  (bvslt  (_ bv1 32) ?B1 ) (bvsle  ?B1 (_ bv15 32) ) ) ) )
(check-sat)
    '''
    fml = And(parse_smt2_string(fml_str))
    print(fml_str)
    print("------After transformation-------")
    tt = aufbv2bv(fml)
    print(to_smt2(tt))


def to_snf(fml):
    tac = Tactic("snf")
    tt = tac.apply(fml).as_expr()
    return tt
    # return to_smt2(tt)


def test3():
    a, b, c = Reals("a b c")
    # fml = Exists(a, ForAll(b, Exists(c, a + b + c > 3)))
    fml1 = Exists(b, ForAll(c, b + c > 3))
    fml2 = Exists(b, b + c > 3)
    fml3 = ForAll(c, Exists(b, b + c > 3))
    fml4 = ForAll(c, b + c > 3)
    fml_5 = Or(ForAll(b, b > 5), ForAll(b, Not(b > 5)))
    # print(to_smt2(fml1))
    # print(to_smt2(Not(fml1)))
    # to_snf(fml_5)
    # to_snf(Not(fml1))
    print(to_snf(Not(fml2)))
    print(to_snf(Not(ForAll(c, fml2))))
    # to_snf(fml3)
    # to_snf(Not(fml3))


test3()
# test2()
