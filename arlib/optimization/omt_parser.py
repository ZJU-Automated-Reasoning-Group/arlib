"""Parse an OMT instance"""

import z3


class OMTParser:
    """Currently, we focus on two modes
    - Single-objective optimization
    - Multi-objective optimization under the boxed mode (each obj is independent)"""

    def __init__(self):
        self.objectives = None
        self.assertions = None

    def parse_string(self, fml_str):
        """FIXME: It sees that Z3 will convert each goal of the form "max f"  to "-f".
            So, should the backend engines regard the converted objectives/goals as all "minimize
            However, many of our queries are of th form "max x; min x; max y; min y; ...."
        """
        s = z3.Optimize()
        s.from_string(fml_str)
        self.assertions = s.assertions()
        self.objectives = s.objectives()
        # for obj in s.objectives():
        #    print("obj: ", obj)
        #    print("decl: ", obj.decl().name())
    def parse_file(self, fml_file_name):
        s = z3.Optimize()
        # s.set("opt.priority", "box")
        s.from_file(fml_file_name)
        self.assertions = s.assertions()
        self.objectives = s.objectives()


# (set-option :opt.priority box)
def demo_omt_parser():
    fml = """
    (declare-const x (_ BitVec 16))
    (declare-const y (_ BitVec 16))
    (assert (bvult x (_ bv100 16)))
    (assert (bvule y (_ bv98 16)))
    (maximize (bvsub x y))
    (minimize (bvadd x y))
    (minimize (bvneg y))
    (check-sat)
    (get-objectives)
    """
    s = OMTParser()
    s.parse_string(fml)


if __name__ == "__main__":
    # a, b, c, d = z3.Ints('a b c d')
    # fml = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))
    demo_omt_parser()
