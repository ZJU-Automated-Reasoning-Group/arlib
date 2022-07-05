#!/usr/bin/python3
"""
Simple portfolio (this servers as a baseline technique)
"""
import argparse
import logging
import multiprocessing
# import random
# import signal
import subprocess

g_process_queue = []


def solve_with_partitioned(formula_file, result_queue):
    print("one worker working")
    cmd = ["/Users/prism/Work/cvc5/build/bin/cvc5", "-q", "--produce-models", formula_file]
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.readlines()
    out = ' '.join([str(element.decode('UTF-8')) for element in out])
    if "unsat" in out:
        ret = "unsat"
    elif "sat" in out:
        ret = "sat"
    else:
        ret = "unknown"
    p.stdout.close()
    # print("ret: ", ret)
    result_queue.put(ret)


def signal_handler(sig, frame):
    global g_process_queue
    try:
        for p in g_process_queue:
            if p: p.terminate()
        logging.debug("processes cleaned!")
    except Exception as e:
        print(e)
        pass


def main():
    # global process_queue

    parser = argparse.ArgumentParser(description="Solve given formula.")
    parser.add_argument("formula", metavar="formula_file", type=str,
                        help="path to .smt2 formula file")

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    parser.add_argument('--workers', dest='workers', default=2, type=int, help="num threads")

    parser.add_argument('--timeout', dest='timeout', default=60, type=int, help="timeout")

    args = parser.parse_args()

    if args.verbose: logging.basicConfig(level=logging.DEBUG)

    formula_file = args.formula

    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()

        n_workers = min(multiprocessing.cpu_count(), int(args.workers))

        for nth in range(n_workers):
            g_process_queue.append(multiprocessing.Process(target=solve_with_partitioned,
                                                           args=(formula_file,
                                                                 result_queue
                                                                 )))

        # Start all
        for p in g_process_queue:
            p.start()

        # Get result
        try:
            # Wait at most 60 seconds for a return
            result = result_queue.get(timeout=int(args.timeout))
        except multiprocessing.queues.Empty:
            result = "unknown"
        # Terminate all
        for p in g_process_queue:
            p.terminate()

    print(result)


if __name__ == "__main__":
    # TODO: if terminated by the caller, we should kill the processes in the process_queue??
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGQUIT, signal_handler)
    # signal.signal(signal.SIGHUP, signal_handler)
    main()
