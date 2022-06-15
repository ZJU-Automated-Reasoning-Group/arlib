# coding: utf-8
# import time
import signal
import sys
import os
import logging
import psutil


def setup_logging(args):
    """
    Initialize logging level
    """
    verbosity_map = {
        -2: logging.ERROR,
        -1: logging.WARN,
        0: logging.CHAT,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.TRACE,
    }
    verbosity = args.verbosity
    logging.getLogger().setLevel(level=verbosity_map.get(verbosity))


def signal_handler(sig, frame):
    """
    Captures the shutdown signals and cleans up all child processes of this process.
    """
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', default='none', type=str, help="file")
    parser.add_argument('--prover', dest='prover', default='seq', type=str, help="The prover for using")
    parser.add_argument('--timeout', dest='timeout', default=8, type=int, help="timeout")
    parser.add_argument('--workers', dest='workers', default=4, type=int, help="workers")
    parser.add_argument('--verbose', dest='verbosity', default=1, type=int, help="verbosity")
    args = parser.parse_args()
    setup_logging(args)

    # Registers signal handler so we can kill all of our child processes.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGQUIT, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
