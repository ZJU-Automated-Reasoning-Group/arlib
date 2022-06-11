from z3 import *
# from pdsmt.pdpllt import main_par
from pdsmt.formula_manager import simple_cdclt


def test():
    x, y, z = z3.BitVecs("x y z", 16)
    fml = z3.And(z3.Or(x > 10, x == 1), z3.Or(x == 5, x == 6))
    s = z3.Solver()
    s.add(fml)
    print(s.to_smt2())


def process_folder(path):
    import os
    filelist = []  # path to smtlib2 files
    for root, dirs, files in os.walk(path):
        for fname in files:
            if os.path.splitext(fname)[1] == '.smt2':
                filelist.append(os.path.join(root, fname))
    print("num files: ", len(filelist))


if __name__ == '__main__':
    # test()
    simple_cdclt("/Users/prism/Work/pdsmt/benchmakrs/simple.smt2")
    # process_folder("/Users/prism/Work/eldarica-bin/tests/z3test/")