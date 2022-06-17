import logging
import time
import z3
from pdsmt.simple_cdclt import simple_cdclt
from pdsmt.parallel_cdclt import parallel_cdclt
from pdsmt.profiler import Profiler, render_profiles


def string_test():
    logging.basicConfig(level=logging.DEBUG)
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
    #with Profiler(True):
    # print(simple_cdclt(fml))
    print(parallel_cdclt(fml, logic="ALL"))
    # print(boolean_abstraction(fml))


def process_file(filename: str, logic: str):
    with open(filename, "r") as f:
        smt2string = f.read()
        # simple_cdclt(smt2string)
        res = parallel_cdclt(smt2string, logic=logic)
        print(res)


if __name__ == '__main__':
    # string_test()
    logging.basicConfig(level=logging.DEBUG)
    # string_test()
    # FIXME: the preprocessor creates a function named "bvsdiv_i", which cannot be recognized by z3??
    # process_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_BV/unsat/bench_4615.smt2", "QF_BV")
    process_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_NRA/unsat/sqrt-1mcosq-8-chunk-0203.smt2", "QF_NRA")


