import z3

from arlib.utils.z3_plus_smtlib_solver import Z3SolverPlus


def test_itp():
    """ Interpolaion with CVC5
    """
    a, b, c = z3.Bools("a b c")
    A = z3.And(a, b, c)
    B = z3.And(a, b)
    # A |= B
    s = Z3SolverPlus()
    # A |= I, I |= B, vars(I) = vars(A) cup vars(B)
    I = s.binary_interpolant(A, B)
    print(I)

    """
    pre = gen_formula_of_logic("QF_LIA")
    post = gen_formula_of_logic("QF_LIA")
    # print(pre)
    pre_vars = [str(var) for var in get_variables(pre)]
    post_vars = [str(var) for var in get_variables(post)]
    if entail(pre, post) and len(list(set(pre_vars).intersection(set(post_vars)))) > 0:
        s = Z3SolverPlus()
        print(s.binary_interpolant(pre, post))  # And(x >= 1)
        return True
    return False
    """


def test_qe():
    """ Compute interpolant with cv5
    """
    x, y, z = z3.Ints("x y z")
    fml = z3.And(x > 1, y > 0)
    qfml = z3.Exists(x, fml)
    s = Z3SolverPlus()
    print(s.qelim(qfml))


def test_abduct():
    """Abduction using cvc5
    """
    a, b, c = z3.Ints("a b c")
    pre = z3.And(b >= c)
    post = b > 10
    s = Z3SolverPlus()
    print(s.abduct(pre, post))  # And(b == c)


def test_sygus():
    """ SyGuS with cvc5
    """
    fun = z3.Function("gle", z3.IntSort(), z3.IntSort(), z3.BoolSort())

    x, y, z = z3.Ints("x y z")

    # s = Solver()
    # s.add(And(fun(x, y), fun(y, z), x < z))
    # s.add(ForAll([x, y], Implies(x > y, fun(x, y))))
    # s.add(ForAll([x, y], Implies(x >= y, fun(x, y))))
    # s.add(ForAll([x, y], Implies(x < y, Not(fun(x, y)))))
    # print(s.check())

    # In SyGus, there are implicit universal quantifiers?
    # cnts_for_max = [fun(x, y) >= x, fun(x, y) >= y, Or(x == fun(x, y), y == fun(x, y))]
    axioms_for_max = [z3.Implies(x > y, fun(x, y)),
                      z3.Implies(x >= y, fun(x, y)),
                      # Not(fun(1, 3)),
                      z3.Implies(x < y, z3.Not(fun(x, y)))
                      ]

    # cnts_for_max = [fun(1, 2) == 2, fun(2, 3) == 3, Or(3 == fun(3, 4), 4 == fun(3, 4))]

    ss = Z3SolverPlus()
    print(ss.sygus([fun], axioms_for_max, [x, y], logic="LIA"))
    # (define-fun max ((x Int) (y Int)) Int (ite (>= (+ x (* (- 1) y)) 0) x y))
    # todo: map the result back to z3; multiple fnctions, etc.


def test_sygus2():
    max2 = z3.Function("max2", z3.BitVecSort(32), z3.BitVecSort(32), z3.BitVecSort(32))
    x, y = z3.BitVecs("x y", 32)
    cnts_for_max = [z3.UGE(max2(x, y), x), z3.UGE(max2(x, y), y), z3.Or(x == max2(x, y), y == max2(x, y))]
    s = Z3SolverPlus()
    print(s.sygus([max2], cnts_for_max, [x, y], logic="BV"))


def test_sygus3():
    """
    multiple functions
    """
    addexpr1 = z3.Function("addexpr1", z3.IntSort(), z3.IntSort(), z3.IntSort())
    addexpr2 = z3.Function("addexpr2", z3.IntSort(), z3.IntSort(), z3.IntSort())
    x, y, z = z3.Ints("x y z")
    cnts = [addexpr1(x, y) + addexpr2(y, x) == x - y]
    s = Z3SolverPlus()
    print(s.sygus([addexpr1, addexpr2], cnts, [x, y], logic="LIA"))

    goal = z3.Function("goal", z3.IntSort(), z3.IntSort(), z3.IntSort(), z3.BoolSort())


def test_sygus_str():
    """
    multiple functions
    """
    ff = z3.Function("ff", z3.StringSort(), z3.StringSort(), z3.StringSort())
    x, y, z = z3.Strings("x y z")
    cnts = [z3.Length(ff(x, y)) >= z3.Length(x), z3.Length(ff(x, y)) >= z3.Length(y)]
    s = Z3SolverPlus()
    print(s.sygus([ff], cnts, [x, y, z], logic="ALL"))


def test():
    cnt = "(assert (>= (+ x ( * (- 1) y)) 0))"
    cnt = "(define-fun gle ((x Int) (y Int)) Bool (>= (+ x (* (- 1) y)) 0))"
    x, y, z = z3.Ints("x y z")
    s = z3.Solver()
    """
    >>> x, y = Ints('x y')
    >>> f = Function('f', IntSort(), IntSort())
    >>> parse_smt2_string('(assert (> (+ foo (g bar)) 0))', decls={ 'foo' : x, 'bar' : y, 'g' : f})
    """
    print(z3.And(z3.parse_smt2_string(cnt, decls={'x': x, 'y': y})))


def test_omt():
    x, y, z, w = z3.Ints("x y z w")
    ss = z3.Optimize()
    ss.from_file("/Users/prism/Work/optimathsat/bin/xx.smt2")
    fml = z3.And(ss.assertions())
    # fml = And(x < 3, y < 8)
    s = Z3SolverPlus()
    # ret = s.optimize(fml, x, minimize=False, logic="QF_LIA")
    # print(ret)
    ret_mut = s.compute_min_max(fml, minimize=[y, z, w], maximize=[y, z, w])
    print(ret_mut)

    # from bvopt.z3opt_util import box_optimize
    # print(box_optimize(fml,  minimize=[x, y], maximize=[x, y]))


def test_all_sat():
    x, y = z3.Ints("x y")
    a, b, c, d = z3.Bools("a b c d")
    fmls = [a == x + y > 0, c == 2 * x + 3 * y < -10, z3.And(z3.Or(a, b), z3.Or(c, d))]
    s = Z3SolverPlus()
    print(s.all_sat(z3.And(fmls), [a, b]))


# test_all_sat()
test_itp()
# test()
# test_abduct()
# test_qe()
# test_sygus()
# test_sygus3()
# test_omt()
