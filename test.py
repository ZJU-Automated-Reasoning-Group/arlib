from z3 import *
from pdsmt.formula_manager import simple_cdclt, boolean_abstraction


def test():
    x, y, z = z3.BitVecs("x y z", 16)
    fml = z3.And(z3.Or(x > 10, x == 1), z3.Or(x == 5, x == 6))
    s = z3.Solver()
    s.add(fml)
    print(s.to_smt2())


def process_file(filename):
    with open(filename, "r") as f:
        smt2string = f.read()
        simple_cdclt(smt2string)


def process_folder(path):
    import os
    file_list = []  # path to smtlib2 files
    for root, dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1] == '.smt2':
                file_list.append(os.path.join(root, fname))
    print("num files: ", len(file_list))
    for file in file_list:
        process_file(file)


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
    print(simple_cdclt(fml))
    # print(boolean_abstraction(fml))


if __name__ == '__main__':
    string_test()
    # process_file("/Users/prism/Work/pdsmt/benchmakrs/simple.smt2")
    # process_folder("/Users/prism/Work/eldarica-bin/tests/z3test/")
