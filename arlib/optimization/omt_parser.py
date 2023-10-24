"""Parse an OMT instance"""

import z3


class OMTParser:

    def __int__(self):
        return

    def parse_file(self, fml_file_name):
        return

    def parse_string(self, fml_str):
        return


def demo_omt_parser():
    fml = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (< x 100))
    (assert (< y 99))
    (maximize x)
    (maximize y)
    (check-sat)
    (get-objectives)
    """
    s = z3.Optimize()
    s.set("opt.priority", "box")
    s.from_string(fml)
    print(s.assertions())
    print(s.objectives())
    # print(s.param_descrs())
    # print(s.)

demo_omt_parser()

