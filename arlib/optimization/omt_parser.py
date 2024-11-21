"""Parse an OMT instance"""

import z3
from z3.z3consts import *


class OMTParser:
    """Currently, we focus on two modes
    1. Single-objective optimization
    2. Multi-objective optimization under the boxed mode (each obj is independent)"""

    def __init__(self):
        """
        For multi-objective optimization,
        """
        self.assertions = None
        self.objectives = []
        self.to_max_obj = True  # convert all objectives to max
        self.to_min_obj = False  # convert all objectives to min
        self.debug = True

    def parse_with_pysmt(self):
        # pysmt does not support
        raise NotImplementedError

    def parse_with_z3(self, fml: str, is_file=False):
        """FIXME: Should we convert all the objectives/goals as all "minimize goals" (as Z3 does)?
            (or should we convert them to "maximize goals"?)
            However, the queries can be of the form "max x; min x; max y; min y; ...."
        """
        s = z3.Optimize()
        if is_file:
            s.from_file(fml)
        else:
            s.from_string(fml)
        self.assertions = s.assertions()
        # We cannot set both self.to_min_obj and self.to_max_obj to True
        assert not (self.to_min_obj and self.to_max_obj)
        if self.to_min_obj:
            # It sees that Z3 will convert each goal of the form "max f"  to "-f".
            # So, we just assign s.objectives() to self.objectives
            self.objectives = s.objectives()
        elif self.to_max_obj:
            # https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml
            # TODO: the semantics of bvneg: [[(bvneg s)]] := nat2bv[m](2^m - bv2nat([[s]]))
            # Z3 will convert each goal of the form "max f"  to "-f".
            # So, we need to "convert them back"?
            for obj in s.objectives():
                # if calling z3.simplify(-obj), the obj may look a bit strange
                if obj.decl().kind() == Z3_OP_BNEG:
                    # self.objectives.append(-obj)
                    # If the obj is of the form "-expr", we can just add "expr" instead of "--expr"?
                    self.objectives.append(obj.children()[0])
                else:
                    self.objectives.append(-obj)
        if self.debug:
            for obj in self.objectives:
                print("obj: ", obj)


# (set-option :opt.priority box)
def demo_omt_parser():
    from arlib.optimization.opt_util import optimize_as_long
    fml_one = """
    (declare-const x (_ BitVec 16)) \n (declare-const y (_ BitVec 16)) \n
    (assert (bvult x (_ bv100 16))) \n (assert (bvule y (_ bv98 16))) \n
    (maximize (bvsub x y)) \n (minimize (bvadd x y)) \n (minimize (bvneg y)) \n (check-sat)
    """
    fml_two = """
    (declare-const x (_ BitVec 4)) \n (declare-const y (_ BitVec 4)) \n
    (assert (bvult x (_ bv5 4))) \n (assert (bvuge y (_ bv3 4))) \n
    (maximize x) \n (minimize x) \n (maximize y) \n (minimize y) \n (check-sat)
    """
    if True:
        x, y = z3.BitVecs("x y", 4)
        fml = z3.And(z3.ULT(x, 5), z3.UGE(y, 3))
        print(optimize_as_long(fml=fml, obj=-x, minimize=False))  # 15?
        # print(optimize_as_long(fml=fml, obj=x, minimize=True))
        # print(optimize_as_long(fml=fml, obj=y, minimize=False))
        # print(optimize_as_long(fml=fml, obj=y, minimize=True))
        # print(box_optimize_as_long(fml, minimize=[], maximize=[x, -x, y, -y]))
    else:
        s = OMTParser()
        s.parse_with_z3(fml_two)
        print(s.objectives)
        # use Z3
        print(box_optimize_as_long(z3.And(s.assertions), minimize=[], maximize=s.objectives)[1])
        # use our implementation
        opt = BVOptimize()
        opt.from_smt_formula(z3.And(s.assertions))
        res = opt.boxed_optimize(goals=s.objectives, is_signed=False)
        print(res)


if __name__ == "__main__":
    # a, b, c, d = z3.Ints('a b c d')
    # fml = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))
    demo_omt_parser()
