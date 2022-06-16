# coding: utf8
import os
import subprocess
from threading import Timer
import logging
# import zlib


def find_smt2_files(path):
    flist = []  # path to smtlib2 files
    for root, dirs, files in os.walk(path):
        for fname in files:
            tt = os.path.splitext(fname)[1]
            if tt == '.smt2' or tt == '.sl':
                flist.append(os.path.join(root, fname))
    return flist


def terminate(process, is_timeout):
    if process.poll() is None:
        try:
            # print("timeout!")
            process.terminate()
            # process.kill()
            is_timeout[0] = True
        except Exception as es:
            # print("error for interrupting")
            print(es)
            pass


def solve_with_bin_solver(cmd, timeout=5):
    """cmd should be a complete cmd"""
    # ret = "unknown"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    is_timeout = [False]
    timer = Timer(timeout, terminate, args=[p, is_timeout])
    timer.start()

    res_out = []
    for line in iter(p.stdout.readline, "b"):
        if line:
            tmp = str(line.decode('UTF-8'))
            res_out.append(tmp)
            print(tmp)
        if p.poll() is None:
            break
    out = ' '.join(res_out)

    # NOTE: the following two lines may lead to strange resource leak
    # out = p.stdout.readlines()
    # out = ' '.join([str(element.decode('UTF-8')) for element in out])
    p.stdout.close()
    timer.cancel()
    if is_timeout[0]:
        return "timeout"
    return out


def solve_file(filename: str, logic: str):
    # cmd = ["/Users/prism/Work/cvc5/build/bin/cvc5", "-q"]
    cmd = ["python3", "/Users/prism/Work/pdsmt/pdsmt.py"]
    cmd.append("--logic")
    cmd.append(logic)
    cmd.append(filename)
    out = solve_with_bin_solver(cmd, 10)
    print(out)


def solve_dir(path: str, logic: str):
    # cmd = ["/Users/prism/Work/cvc5/build/bin/cvc5", "-q"]
    files = find_smt2_files(path)
    print(len(files))
    processed = 1
    for file in files:
        print("Solving: ", file)
        solve_file(file, logic)
        processed += 1
        if processed >= 150:
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    solve_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_LIA/sat/unbd-sage13.smt2", "QF_LIA")
    # solve_dir("/Users/prism/Work/semantic-fusion-seeds-master/QF_BV", "QF_BV")
    # solve_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_BV/unsat/bench_1987.smt2", "QF_BV")

"""
FIXME:
1.  The preprocessor creates a function named "bvsdiv_i", which cannot be recognized by z3??
    solve_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_BV/unsat/bench_4615.smt2", "QF_BV")

2.  Only one sampled Boolean model, which is hard for the theory solver to decide
    solve_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_NRA/unsat/sqrt-1mcosq-8-chunk-0203.smt2", "QF_NRA")

     # the one below can be solved via smt.arith.solver=2 (the default one is 6, which was recently introduced)
     # NOTE: timeout does not work for it (triggering a strange issue)
    solve_file("/Users/prism/Work/semantic-fusion-seeds-master/QF_LIA/sat/unbd-sage13.smt2", "QF_LIA")

3. 
"""