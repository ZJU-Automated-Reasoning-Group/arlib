# coding: utf-8
# import time
import psutil
import signal
from z3 import *

"""
Parallel SMT Solving
"""


def signal_handler(sig, frame):
    """
    Captures the shutdown signals and cleans up all child processes of this process.
    """
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()
    sys.exit(0)


def solve_smt_file(filename: str, prover="all"):

    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', default='none', type=str, help="file")
    parser.add_argument('--prover', dest='prover', default='seq', type=str, help="The prover for using")
    parser.add_argument('--timeout', dest='timeout', default=8, type=int, help="timeout")
    parser.add_argument('--workers', dest='workers', default=4, type=int, help="workers")
    args = parser.parse_args()

    # Registers signal handler so we can kill all of our child processes.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    solve_smt_file(args.file, args.prover)
