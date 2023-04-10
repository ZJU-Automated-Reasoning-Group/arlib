# coding: utf-8
# pylint: disable=too-few-public-methods
# import time
"""
The entrance of the parallel (and distributed) SMT solving engine
"""
import os
import signal
import psutil

from arlib.cdclt.cdclt_solver import ParallelCDCLTSolver

G_ARGS = None


def signal_handler(sig, frame):
    """
    Handles the shutdown signals and cleans up all child processes of this process.
    TODO: is this correct?
    Args:
        sig (int): The signal number received.
        frame (frame): The current stack frame.
    """
    # print("handling signals")
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()


def process_file(filename: str, logic: str, mode: str):
    """
    Processes the given SMT2 file and solves it using the specified logic.

    Args:
        filename (str): The path to the SMT2 file to be processed.
        logic (str): The logic to be used for solving the SMT2 file.
    """
    # g_smt2_file = open(filename, "r")
    # smt2string = g_smt2_file.read()
    # simple_cdclt(smt2string)
    sol = ParallelCDCLTSolver(mode=mode)
    ret = sol.solve_smt2_file(filename=filename, logic=logic)
    print(ret)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', dest='timeout', default=8, type=int, help="timeout")
    parser.add_argument('--workers', dest='workers', default=4, type=int, help="workers")
    parser.add_argument('--verbose', dest='verbosity', default=1, type=int, help="verbosity")
    parser.add_argument('--logic', dest='logic', default='ALL', type=str, help="logic to use")
    parser.add_argument('--mode', dest='mode', default="process", type=str,
                        help="the mode:"
                             "process: process-based  "
                             "thread: thread-based  "
                             "preprocess: Dump the Boolean skeleton after pre-processing"
                             " (as a CNF/DIMACS file), and the name wile infile.cnf")
    parser.add_argument('infile', help='the input file (in SMT-LIB v2 format)')
    G_ARGS = parser.parse_args()

    # Which signals should we?
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Registers signal handler, so we can kill all of our child processes.
    process_file(G_ARGS.infile, G_ARGS.logic, G_ARGS.mode)
