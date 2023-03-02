import logging
from arlib.cdcl.parallel_cdclt_process import parallel_cdclt_process
from arlib.tests import TestCase, main


def string_test():
    fml = """
    (set-info :status unknown)
(declare-fun x () Int)
(declare-fun z () Int)
(declare-fun w () Int)
(declare-fun y () Int)
(assert
 (let (($x37 (> z x)))
 (let (($x10 (= (- 62) w)))
 (let (($x63 (< 93 x)))
 (let (($x62 (and (distinct 55 y) true)))
 (let (($x23 (not $x62)))
 (let (($x94 (and (distinct 93 z) true)))
 (let (($x51 (or $x94 $x63)))
 (let (($x66 (and $x62 $x51)))
 (and (or $x23 $x23 $x94 $x51 $x66) (or $x66 $x37 $x94 $x51 $x62 $x10 $x63 $x23) (or $x10 $x37 $x23 $x94 $x63 $x51 $x66 $x23) (or $x23 $x23 $x37 $x66 $x10 $x62) (or $x10 $x63 $x23 $x94) (or $x62 $x23 $x63 $x66 $x23 $x94 $x37) (or $x10 $x63 $x37 $x51) $x94 (or $x23 $x63 $x10 $x37)))))))))))
(check-sat)
    """
    # with Profiler(True):
    # print(simple_cdclt(fml))
    print(parallel_cdclt_process(fml, logic="ALL"))


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
    logger = logging.getLogger('tmp')
    logger.setLevel(logging.DEBUG)

    main()
