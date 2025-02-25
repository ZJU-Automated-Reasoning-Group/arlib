"""Utils for manipulating values"""

import re
from typing import List
import itertools
import z3
from z3 import BitVecVal, Concat, Extract

RE_GET_EXPR_VALUE_ALL = re.compile(
    r"\(([a-zA-Z0-9_]*)[ \n\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false)\)"
    r"\((p@[0-9]*)[ \n\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false)\)"
)


def powerset(elements: List):
    """Generates the powerset of the given elements set.

    E.g., powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    return itertools.chain.from_iterable(itertools.combinations(elements, r)
                                         for r in range(len(elements) + 1))


# Bit Operations
def set_bit(v: int, index: int, x: int):
    """Set the index:th bit of v to x, and return the new value."""
    mask = 1 << index
    if x:
        v |= mask
    else:
        v &= ~mask
    return v


def twos_complement(val: int, bits: int):
    """Returns the 2-complemented value of val assuming bits word width"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set
        val = val - (2 ** bits)  # compute negative value
    return val


def convert_smtlib_models_to_python_value(v):
    """
    For converting SMT-LIB models to Python values
    TODO: we may need to deal with other types of variables, e.g., int, real, string, etc.
    """
    r = None
    if v == "true":
        r = True
    elif v == "false":
        r = False
    elif v.startswith("#b"):
        r = int(v[2:], 2)
    elif v.startswith("#x"):
        r = int(v[2:], 16)
    elif v.startswith("_ bv"):
        r = int(v[len("_ bv"): -len(" 256")], 10)
    elif v.startswith("(_ bv"):
        v = v[len("(_ bv"):]
        r = int(v[: v.find(" ")], 10)

    assert r is not None
    return r


def zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to 0.
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = z3.Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def one_extension(formula: z3.BitVecRef, bit_places: int):
    """Set the rest of bits on the left to 1.
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to the value of the sign bit.
    """
    sign_bit = Extract(bit_places - 1, bit_places - 1, formula)

    complement = sign_bit
    for _ in range(formula.size() - bit_places - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def right_zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 0.
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)
    return formula


def right_one_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 1.
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)
    return formula


def right_sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to the value of the sign bit.
    """
    sign_bit_position = formula.size() - bit_places
    sign_bit = Extract(sign_bit_position, sign_bit_position, formula)

    complement = sign_bit
    for _ in range(sign_bit_position - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(Extract(formula.size() - 1,
                             sign_bit_position,
                             formula),
                     complement)
    return formula


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

    