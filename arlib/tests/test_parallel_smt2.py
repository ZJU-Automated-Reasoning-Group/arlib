import logging
from arlib.smt.pcdclt.parallel_cdclt_process import parallel_cdclt_process
from arlib.smt.pcdclt.parallel_cdclt_process_new import parallel_cdclt_process_new
from arlib.tests import TestCase, main


class TestParallelSMTSolver2(TestCase):

    def test_par_solver2(self):
        return
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
        # print(parallel_cdclt_process(fml, logic="ALL"))
        print(parallel_cdclt_process_new(fml, logic="ALL"))


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    log = logging.getLogger('custom logger')
    log.setLevel(logging.DEBUG)

    main()
