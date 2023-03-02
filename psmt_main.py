# coding: utf-8
# pylint: disable=too-few-public-methods
# import time
"""
The entrance of the parallel (and distributed) SMT solving engine
"""
import os
import signal
import psutil

from arlib.cdcl.cdclt_solver import ParallelCDCLSolver

g_smt2_file = None


def signal_handler(sig, frame):
    """Captures the shutdown signals and cleans up all child processes of this process."""
    # print("handling signals")
    if g_smt2_file:
        g_smt2_file.close()
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()


def process_file(filename: str, logic: str):
    global g_smt2_file
    g_smt2_file = open(filename, "r")
    smt2string = g_smt2_file.read()
    # simple_cdclt(smt2string)
    sol = ParallelCDCLSolver()
    ret = sol.solve_smt2_string(smt2string, logic)
    print(ret)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--prover', dest='prover', default='seq', type=str, help="The prover for using")
    parser.add_argument('--timeout', dest='timeout', default=8, type=int, help="timeout")
    parser.add_argument('--workers', dest='workers', default=4, type=int, help="workers")
    parser.add_argument('--verbose', dest='verbosity', default=1, type=int, help="verbosity")
    parser.add_argument('--logic', dest='logic', default='ALL', type=str, help="logic to use")
    parser.add_argument('infile', help='the input file (in SMT-LIB v2 format)')
    args = parser.parse_args()

    # Which signals should we?
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Registers signal handler, so we can kill all of our child processes.
    process_file(args.infile, args.logic)
