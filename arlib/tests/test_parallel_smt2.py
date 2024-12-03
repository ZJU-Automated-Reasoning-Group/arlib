import logging
from arlib.smt.pcdclt import parallel_cdclt_process
from arlib.tests import TestCase, main


def string_test():
    fml = """
(set-logic QF_LIA)
(declare-fun x () Int)
(declare-fun y () Int)
(assert (or (= x 1) (> x 2)))
(assert (or (= y 3) (> y 4)))
(assert (= 4 (+ x y)))
(check-sat)
    """
    # with Profiler(True):
    # print(simple_cdclt(fml))
    return
    # print(parallel_cdclt_process(fml, logic="ALL"))


class TestParallelSMTSolver2(TestCase):

    def test_par_solver2(self):
        string_test()


def solve_file(filename: str, logic: str):
    with open(filename, "r") as f:
        smt2string = f.read()
        # simple_cdclt(smt2string)
        res = parallel_cdclt_process(smt2string, logic=logic)
        print(res)


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    log = logging.getLogger('custom logger')
    log.setLevel(logging.DEBUG)

    main()
