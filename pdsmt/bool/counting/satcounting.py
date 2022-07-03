"""
Counting solutions of a CNF formula
"""
import subprocess
from threading import Timer
from typing import List


def terminate(process, is_timeout: List):
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            pass


def solve_with_bin_solver(cmd, timeout=30):
    """
    cmd should be a complete cmd
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    is_timeout = [False]
    timer = Timer(timeout, terminate, args=[p, is_timeout])
    timer.start()
    out = p.stdout.readlines()
    out = ' '.join([str(element.decode('UTF-8')) for element in out])
    p.stdout.close()
    timer.cancel()
    if p.poll() is None:
        p.terminate() # need this?

    if is_timeout[0]:
        return "timeout"
    return out


def count_models_by_sharp_sat(sharp_bin: str, cnf_file: str, timeout: int) -> str:
    """
    :param sharp_bin: the path of sharpSAT
    :param cnf_file: the CNF file
    :param timeout: time limit for the counting (in seconds)
    :return:
    """
    assert sharp_bin != ""
    cmd = [sharp_bin, cnf_file]
    return solve_with_bin_solver(cmd, timeout)


class SATModelCounter:
    """
    Model counter and SAT formula
    """

    def __init__(self):
        self.strategy = "sharpSAT"
        self.timeout = 30
        self.sharp_sat_bin = ""

    def count_models(self, cnt_file: str):
        assert self.sharp_sat_bin != ""
        return count_models_by_sharp_sat(self.sharp_sat_bin, cnt_file, self.timeout)

    def cube_and_count(self, cnf_file: str):
        """
        1. Generate a set of disjoint cubes such that they can be extended to be models of the formula
            C1: a, b
            C2: Not(a), b
            C3: a, Not b
            C4: Not(a), Not(b)
        2. Count the models subject to each cube in parallel
        """
        raise NotImplementedError

