"""
Interface for QBF solving
"""

import z3
from z3 import is_true, BoolVal
import functools

foldr = lambda func, acc, xs: functools.reduce(lambda x, y: func(y, x), xs[::-1], acc)


def z3_val_to_int(z3_val):
    return 1 if is_true(z3_val) else 0


def int_vec_to_z3(int_vec):
    return [BoolVal(val == 1) for val in int_vec]


q_to_z3 = {1: z3.ForAll, -1: z3.Exists}


class QBF:

    def __init__(self, prop_formula, q_list=None):
        super(QBF, self).__init__()
        if q_list is None:
            q_list = []
        self._q_list = q_list
        self._prop = prop_formula

    def get_prop(self):
        return self._prop

    def get_q_list(self):
        return self._q_list

    def to_z3(self):
        return foldr(lambda q_v, f: q_to_z3[q_v[0]](q_v[1], f), self._prop, self._q_list)

    def negate(self):
        new_q_list = [(-_q, _v) for (_q, _v) in self._q_list]
        return QBF(self._prop.children()[0] if z3.is_not(self._prop) else z3.Not(self._prop), new_q_list)

    def well_named(self):
        """
    q_list = self.get_q_list()
        appeared = set()
        for _, var_vec in q_list:
            for _v in var_vec:
                if _v in appeared:
                    return False
                appeared.add(_v)
        return True
    """
        return True

    def __eq__(self, o):
        return self._prop.eq(o.get_prop()) and self._q_list == o.get_q_list()

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash((hash(self._prop), hash(tuple(self._q_list))))


def test():
    prop_formula = z3.And(z3.Bool('x'), z3.Bool('y'))
    q_list = [(1, ['x']), (-1, ['y'])]
    qbf = QBF(prop_formula, q_list)


test()
