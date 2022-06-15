# coding: utf-8
# import time
import signal
import psutil
import sys
import os
import logging


def setup_logging(args):
    logging.CHAT = 25
    logging.addLevelName(logging.CHAT, "CHAT")
    logging.chat = lambda msg, *args, **kwargs: logging.log(
        logging.CHAT, msg, *args, **kwargs)
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, "TRACE")
    logging.trace = lambda msg, *args, **kwargs: logging.log(
        logging.TRACE, msg, *args, **kwargs)
    logging.basicConfig(format='[ddSMT %(levelname)s] %(message)s')
    verbositymap = {
        -2: logging.ERROR,
        -1: logging.WARN,
        0: logging.CHAT,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.TRACE,
    }
    verbosity = args.verbosity
    logging.getLogger().setLevel(
        level=verbositymap.get(verbosity, logging.DEBUG))


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
