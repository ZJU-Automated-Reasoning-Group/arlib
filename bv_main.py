"""
The entrance of the bit-vector solving engine
"""
import os
import signal
import psutil
import logging
from arlib.bv.bv_sat import BVSolver


def signal_handler(sig, frame):
    """Captures the shutdown signals and cleans up all child processes of this process?"""
    # print("handling signals")
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()


def process_file(filename: str):
    sol = BVSolver()
    sol.from_smt_file(filename)
    print(sol.check_sat())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', dest='timeout', default=8, type=int, help="timeout")
    parser.add_argument('--verbose', dest='verbosity', default=1, type=int, help="verbosity level")
    parser.add_argument('--model', dest='model', default=True, type=bool, help="enable model generation or not")
    parser.add_argument('--unsat_core', dest='unsat_core', default=False, type=bool,
                        help="enable core generation or not")
    parser.add_argument('--sat_engine', dest='sat_engine', default=1, type=int,
                        help="sat engines: 0: z3 (use the interval sat engine of z3) "
                             "1: pysat (TBD, as it supports several engines), "
                             "2: binary solver (allow the user to specify a path for bin solvers)")
    parser.add_argument('infile', help='the input file (in SMT-LIB v2 format)')
    args = parser.parse_args()

    if args.verbosity == 2:
        logging.basicConfig(level=logging.DEBUG)

    # Which signals should we?
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Registers signal handler, so we can kill all of our child processes.
    process_file(args.infile)
