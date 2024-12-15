"""
Some APIs/functions for playing with Z3 exper

- absolute_value_bv
- absolute_value_int
- get_variables
- get_atoms
- to_smtlib2
- is_function_symbol
- get_function_symbols
- skolemize
- big_and
- big_or
- negate
- is_expr_var
- is_expr_val
- is_term
- is_atom
- is_pos_lit
- is_neg_lit
- is_lit

- create_function_body_str
- z3_string_decoder
- z3_value_to_python
- get_z3_logic
"""

from typing import List, Set
import z3
from z3.z3util import get_vars


def absolute_value_bv(bv: z3.BitVecRef):
    """
    Based on: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
    Operation:
        Desired behavior is by definition (bv < 0) ? -bv : bv
        Now let mask := bv >> (bv.size() - 1)
        Note because of sign extension:
            bv >> (bv.size() - 1) == (bv < 0) ? -1 : 0
        Recall:
            -x == ~x + 1 => ~(x - 1) == -(x - 1) -1 == -x
            ~x == -1^x
             x ==  0^x
        now if bv < 0 then -bv == -1^(bv - 1) == mask ^ (bv + mask)
        else bv == 0^(bv + 0) == mask^(bv + mask)
        hence for all bv, absolute_value(bv) == mask ^ (bv + mask)
    """
    mask = bv >> (bv.size() - 1)
    return mask ^ (bv + mask)


def absolute_value_int(val):
    """
    Absolute value for integer encoding
    """
    return z3.If(val >= 0, val, -val)


def get_expr_vars(exp):
    """z3.z3util.get_vars can be very slow; so we use the
    cutomized version"""
    try:
        syms = set()
        stack = [exp]

        while stack:
            e = stack.pop()
            if z3.is_app(e):
                if e.num_args() == 0 and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    syms.add(e)
                else:
                    stack.extend(e.children())

        return list(syms)
    except z3.Z3Exception as ex:
        print(ex)
        return False


def get_variables(exp: z3.ExprRef) -> [z3.ExprRef]:
    """Get variables of exp"""
    # return get_vars(exp)  # this can be very slow
    return get_expr_vars(exp)


def get_atoms(expr: z3.BoolRef):
    """
    Get all atomic predicates in a formula
    """
    a_set = set()

    def get_preds_(exp):
        if exp in a_set:
            return
        if z3.is_not(exp):
            a_set.add(exp)
        if z3.is_and(exp) or z3.is_or(exp):
            for e_ in exp.children():
                get_preds_(e_)
            return
        assert (z3.is_bool(exp))
        a_set.add(exp)

    # convert to NNF and then look for preds
    exp = z3.Tactic('nnf')(expr).as_expr()
    get_preds_(exp)
    return a_set


def to_smtlib2(expr: z3.BoolRef) -> str:
    """"
    To SMT-LIB2 string
    """
    sol = z3.Solver()
    sol.add(expr)
    return sol.to_smt2()


def is_function_symbol(exp: z3.ExprRef) -> bool:
    """Decide"""
    if not z3.is_app(exp):
        return False
    if z3.is_const(exp):
        return False

    func = exp.decl()
    if func.range() == z3.BoolSort():
        # predicate symbol
        return False

    if func.name().lower() == 'if':
        return False

    return True


def get_function_symbols(exp: z3.ExprRef) -> Set[z3.FuncDeclRef]:
    """Find function symbols in a Z3 expr"""
    fsymbols = set()
    if is_function_symbol(exp):
        fsymbols.add(exp.decl())

    for child in exp.children():
        fsymbols.update(get_function_symbols(child))

    return fsymbols


def skolemize(exp: z3.ExprRef) -> z3.ExprRef:
    """To Skolem normal form? (How about snf)"""
    goal = z3.Goal()
    goal.add(exp)
    tactic = z3.Tactic('snf')
    res = tactic(goal)
    return res.as_expr()


def big_and(exp_list: List[z3.ExprRef]):
    """make and"""
    if len(exp_list) == 1:
        return exp_list[0]
    return z3.And(*exp_list)


def big_or(list_of_exp: List[z3.ExprRef]):
    if len(list_of_exp) == 1:
        return list_of_exp[0]
    return z3.Or(*list_of_exp)


def negate(fml: z3.ExprRef) -> z3.ExprRef:
    """Negate a formula"""
    if z3.is_not(fml):
        return fml.arg(0)
    return z3.Not(fml)


def ctx_simplify(exp: z3.ExprRef):
    """Perform complex simplifications (can be slow)"""
    return z3.Tactic('ctx-solver-simplify')(exp).as_expr()


def is_expr_var(exp) -> bool:
    """
    Check if a is a variable. E.g. x is a var but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert is_expr_var(Int('7'))
    >>> assert not is_expr_var(IntVal('7'))
    >>> assert is_expr_var(Bool('y'))
    >>> assert not is_expr_var(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_var(LOnOff)
    >>> assert not is_expr_var(On)
    >>> assert is_expr_var(Block)
    >>> assert is_expr_var(SafetyInjection)
    """

    return z3.is_const(exp) and exp.decl().kind() == z3.Z3_OP_UNINTERPRETED


def is_expr_val(exp) -> bool:
    """
    Check if the input formula is a value. E.g. 3 is a value but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert not is_expr_val(Int('7'))
    >>> assert is_expr_val(IntVal('7'))
    >>> assert not is_expr_val(Bool('y'))
    >>> assert not is_expr_val(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_val(LOnOff)
    >>> assert is_expr_val(On)
    >>> assert not is_expr_val(Block)
    >>> assert not is_expr_val(SafetyInjection)
    """
    return z3.is_const(exp) and exp.decl().kind() != z3.Z3_OP_UNINTERPRETED


def is_term(exp) -> bool:
    """
    Check if the input formula is a term. In FOL, terms are
    defined as term := const | var | f(t1,...,tn) where ti are terms.
    Examples:
    >>> from z3 import *
    >>> assert is_term(Bool('x'))
    >>> assert not is_term(And(Bool('x'),Bool('y')))
    >>> assert not is_term(And(Bool('x'),Not(Bool('y'))))
    >>> assert is_term(IntVal(3))
    >>> assert is_term(Int('x'))
    >>> assert is_term(Int('x') + Int('y'))
    >>> assert not is_term(Int('x') + Int('y') > 3)
    >>> assert not is_term(And(Int('x')==0,Int('y')==3))
    >>> assert not is_term(Int('x')==0)
    >>> assert not is_term(3)
    >>> assert not is_term(Bool('x') == (Int('y')==Int('z')))
    """

    if not z3.is_expr(exp):
        return False
    if z3.is_const(exp):  # covers both const value and var
        return True
    return not z3.is_bool(exp) and all(is_term(c) for c in exp.children())


CONNECTIVE_OPS = [z3.Z3_OP_NOT, z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_IMPLIES,
                  z3.Z3_OP_IFF, z3.Z3_OP_ITE, z3.Z3_OP_XOR]


def is_atom(exp) -> bool:
    """
    Check if the input formula is an atom. In FOL, atoms are
    defined as atom := t1 = t2 | R(t1,..,tn) where ti are terms.
    In addition, this function also allows Bool variable to
    be terms (in propositional logic, a bool variable is considered term)
    """
    if not z3.is_bool(exp):
        return False

    if is_expr_val(exp):
        return False

    if is_expr_var(exp):
        return True

    return z3.is_app(exp) and exp.decl().kind() not in CONNECTIVE_OPS and \
        all(is_term(c) for c in exp.children())


def is_pos_lit(fml) -> bool:
    """
    Check if the input formula is a positive literal,  i.e. an atom
    >>> is_pos_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_atom(fml)


def is_neg_lit(exp) -> bool:
    """
    Check if the input formula is a negative literal
    EXAMPLES:
    >>> from z3 import *
    >>> is_term(3)
    False
    >>> is_neg_lit(Not(Bool('x')))
    True
    >>> is_neg_lit(Not(BoolVal(False)))
    False
    >>> is_neg_lit(BoolVal(True))
    False
    >>> is_neg_lit(BoolVal(False))
    False
    >>> is_neg_lit(Not(Int('x') + Int('y') > 3))
    True
    >>> is_neg_lit(Not(Bool('x') == BoolVal(True)))
    True
    >>> is_neg_lit(Not(Int('x') == 3))
    True
    >>> is_neg_lit(Not(BoolVal(True)))
    False
    """
    return z3.is_not(exp) and is_pos_lit(exp.children()[0])


def is_lit(exp) -> bool:
    """
    Check if the input formula is a negative literal
    >>> is_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_pos_lit(exp) or is_neg_lit(exp)


def create_function_body_str(funcname: str, varlist: List, body: z3.ExprRef) -> [str]:
    """
    Creates a string representation of a function body which can be used to define a new function in SMT-LIB2 format.

    Parameters:
    -----------
    funcname : str
        The name of the function to be defined.
    varlist : List
        The list of input variables for the function.
    body : z3.ExprRef
        The body of the function as a Z3 expression.

    Returns:
    -------
    str
        A string representation of the function body in SMT-LIB2 format.
    """
    res = []
    target = "(define-fun {} (".format(funcname)
    for i in range(len(varlist)):
        target += "({} {}) ".format(str(varlist[i]), varlist[i].sort().sexpr())
    target += ") Bool {})".format(body.sexpr())  # return value
    res.append(target)

    for var in varlist:
        res.append("(declare-const {} {})".format(var, var.sort().sexpr()))
    return res


def z3_string_decoder(z3str: z3.StringVal) -> str:
    """Convert z3.StringVal to python string"""
    length = z3.Int("length")
    tmp_string = z3.String("ts")
    solver = z3.Solver()
    solver.add(tmp_string == z3str)
    solver.add(z3.Length(tmp_string) == length)
    assert solver.check() == z3.sat

    model = solver.model()
    assert model[length].is_int()
    num_chars = model[length].as_long()

    solver.push()
    char_bvs = []
    for i in range(num_chars):
        char_bvs.append(z3.BitVec("ch_%d" % i, 8))
        solver.add(z3.Unit(char_bvs[i]) == z3.SubString(tmp_string, i, 1))

    assert solver.check() == z3.sat
    model = solver.model()
    python_string = "".join([chr(model[ch].as_long()) for ch in char_bvs])
    return python_string


def z3_value_to_python(value):
    if z3.is_true(value):
        return True
    elif z3.is_false(value):
        return False
    elif z3.is_int_value(value):
        return value.as_long()
    elif z3.is_rational_value(value):
        return float(value.numerator_as_long()) / float(value.denominator_as_long())
    elif z3.is_string_value(value):
        return z3_string_decoder(value)
    elif z3.is_algebraic_value(value):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


class FormulaInfo:
    """For formula info"""

    def __init__(self, fml):
        self.formula = fml
        self.has_quantifier = self.has_quantifier()
        self.logic = self.get_logic()

    def apply_probe(self, name):
        goal = z3.Goal()
        goal.add(self.formula)
        probe = z3.Probe(name)
        return probe(goal)

    def has_quantifier(self):
        return self.apply_probe('has-quantifiers')

    def logic_has_bv(self):
        return "BV" in self.logic

    def get_logic(self):
        """
        TODO: how about string, array, and FP?
        """
        try:
            if not self.has_quantifier:
                if self.apply_probe("is-propositional"):
                    return "QF_UF"
                if self.apply_probe("is-qfbv"):
                    return "QF_BV"
                if self.apply_probe("is-qfaufbv"):
                    return "QF_AUFBV"
                if self.apply_probe("is-qflia"):
                    return "QF_LIA"
                # elif self.apply_probe("is-quauflia"):
                #    return "QF_AUFLIA"
                if self.apply_probe("is-qflra"):
                    return "QF_LRA"
                if self.apply_probe("is-qflira"):
                    return "QF_LIRA"
                if self.apply_probe("is-qfnia"):
                    return "QF_NIA"
                if self.apply_probe("is-qfnra"):
                    return "QF_NRA"
                if self.apply_probe("is-qfufnra"):
                    return "QF_UFNRA"
                return "ALL"
            else:
                if self.apply_probe("is-lia"):
                    return "LIA"
                if self.apply_probe("is-lra"):
                    return "LRA"
                if self.apply_probe("is-lira"):
                    return "LIRA"
                if self.apply_probe("is-nia"):
                    return "NIA"
                if self.apply_probe("is-nra"):
                    return "NRA"
                if self.apply_probe("is-nira"):
                    return "NIRA"
                return "ALL"
        except Exception as ex:
            print(ex)
            return "ALL"


def get_z3_logic(fml: z3.ExprRef):
    fml_info = FormulaInfo(fml)
    return fml_info.get_logic()


def eval_predicates(model: z3.ModelRef, predicates: List[z3.BoolRef]):
    """ Let m be a model of a formula phi, preds be a set of predicates
    """
    res = []
    for pred in predicates:
        if z3.is_true(model.eval(pred)):
            res.append(pred)
        elif z3.is_false(model.eval(pred)):
            res.append(negate(pred))
        else:
            pass
    return res
