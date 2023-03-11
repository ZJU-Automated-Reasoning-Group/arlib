"""
Model counting for DIMACS files
"""
import os
import subprocess
from typing import List
from threading import Timer
import logging
from pathlib import Path
import uuid

project_root_dir = str(Path(__file__).parent.parent.parent.parent)
sharp_sat_bin = project_root_dir + "/bin_solvers/sharpSAT"
sharp_sat_timeout = 300

logger = logging.getLogger(__name__)

def terminate(process, is_timeout: List):
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            print(ex)


def write_dimacs_to_file(header, clauses, output_file: str):
    # print("header: ", header)
    # print("clauses: ", len(clauses), clauses)
    with open(output_file, 'w+') as file:
        for info in header:
            file.write(info + "\n")
        for cls in clauses:
            file.write(cls + " 0\n")

def count_dimacs_solutions(header: List, str_clauses: List):
    solutions = 0
    output_file = '/tmp/{}.cnf'.format(str(uuid.uuid1()))
    write_dimacs_to_file(header, str_clauses, output_file)
    cmd = [sharp_sat_bin, output_file]
    print("Calling sharpSAT")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(sharp_sat_timeout, terminate, args=[p, is_timeout])
    timer.start()
    try:
        find_sol_line = False
        for line in iter(p.stdout.readline, ''):
            if not line: break
            decode_line = line.decode('UTF-8')
            if find_sol_line:
                # print("sharpSAT res: ", decode_line)
                solutions = int(decode_line)
                break
            if "solutions" in decode_line:
                find_sol_line = True
    except Exception as ex:
        print(ex)
        print("exception when running sharpSAT, will return false")
        if os.path.isfile(output_file):
            os.remove(output_file)
    if is_timeout[0]: logging.debug("sharpSAT timeout")  # should we put it in the above try scope?
    p.stdout.close()  # close?
    timer.cancel()
    if p.poll() is None:
        p.terminate()
    # process time is not current (it seems to miss the time spent on sharpSAT
    # print("Time:", counting_timer() - time_start)
    if os.path.isfile(output_file):
        os.remove(output_file)
    return solutions


def cube_and_conquer_sharp_sat(header: List, str_clauses: List):
    """
    1. Generate a set of disjoint cubes such that they can be extended to be models of the formula
        C1: a, b
        C2: Not(a), b
        C3: a, Not b
        C4: Not(a), Not(b)
    2. Count the models subject to each cube in parallel
    """
    solutions = 0
    return solutions
