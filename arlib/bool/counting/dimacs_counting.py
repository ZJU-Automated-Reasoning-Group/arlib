"""
Model counting for DIMACS files
"""
import copy
import os
import subprocess
from typing import List
from threading import Timer
import logging
import uuid

import multiprocessing
from multiprocessing import cpu_count

from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.bool.pysat_cnf import gen_cubes

from arlib.global_params import global_config

sharp_sat_timeout = 600

logger = logging.getLogger(__name__)


def terminate(process, is_timeout: List):
    """
    Terminate a given process and set the is_timeout flag to True.

    Args:
        process (subprocess.Popen): The process to be terminated.
        is_timeout (List): A list containing a single boolean value indicating if the process has timed out.
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            print(ex)


def write_dimacs_to_file(header: List[str], clauses: List[str], output_file: str):
    """
    Write the header and clauses of a DIMACS CNF formula to a file.

    Args:
        header (List[str]): A list containing the header information of the DIMACS CNF file.
        clauses (List[str]): A list of strings representing the clauses of the DIMACS CNF file.
        output_file (str): The path to the output file where the DIMACS CNF formula will be written.
    """
    # print("header: ", header)
    # print("clauses: ", len(clauses), clauses)
    with open(output_file, 'w+') as file:
        for info in header:
            file.write(info + "\n")
        for cls in clauses:
            file.write(cls + " 0\n")


def call_sharp_sat(cnf_filename: str):
    """
    Call the sharpSAT solver on a given DIMACS CNF file and return the number of solutions.

    Args:
        cnf_filename (str): The path to the DIMACS CNF file.

    Returns:
        int: The number of solutions for the given DIMACS CNF formula.
    """

    solutions = -1
    cmd = [global_config.get_solver_path("sharp_sat"), cnf_filename]
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
        if os.path.isfile(cnf_filename):
            os.remove(cnf_filename)
    if is_timeout[0]: logging.debug("sharpSAT timeout")  # should we put it in the above try scope?
    p.stdout.close()  # close?
    timer.cancel()
    if p.poll() is None:
        p.terminate()
    # process time is not current (it seems to miss the time spent on sharpSAT
    # print("Time:", counting_timer() - time_start)
    if os.path.isfile(cnf_filename):
        os.remove(cnf_filename)
    return solutions


def count_dimacs_solutions(header: List, str_clauses: List):
    """
    Count the number of solutions for a given DIMACS CNF formula.

    Args:
        header (List): A list containing the header information of the DIMACS CNF file.
        str_clauses (List): A list of strings representing the clauses of the DIMACS CNF file.

    Returns:
        int: The number of solutions for the given DIMACS CNF formula.
    """
    output_file = '/tmp/{}.cnf'.format(str(uuid.uuid1()))
    write_dimacs_to_file(header, str_clauses, output_file)
    return call_sharp_sat(output_file)


def check_sat(clauses, assumptions):
    """
    Check the satisfiability of a CNF formula with given assumptions.

    Args:
        clauses (List): A list of clauses representing the CNF formula.
        assumptions (List): A list of literals representing the assumptions.

    Returns:
        bool: True if the CNF formula is satisfiable under the given assumptions, False otherwise.
    """
    solver = Solver(bootstrap_with=clauses)
    ans = solver.solve(assumptions=assumptions)
    return ans


def count_dimacs_solutions_parallel(header: List[str], clauses: List[str]):
    """
    Count the number of solutions for a given DIMACS CNF formula in parallel.
       1. Generate a set of disjoint cubes such that they can be extended to be models of the formula
        C1: a, b
        C2: Not(a), b
        C3: a, Not b
        C4: Not(a), Not(b)
      2. Count the models subject to each cube in parallel, and sum the results

    Args:
        header (List[str]): A list containing the header information of the DIMACS CNF file.
        clauses (List[str]): A list of strings representing the clauses of the DIMACS CNF file.

    Returns:
        int: The number of solutions for the given DIMACS CNF formula.
    """

    dimacs_str = ""
    for info in header:
        dimacs_str += "{}\n".format(info)
    for cls in clauses:
        dimacs_str += "{} 0\n".format(cls)

    cnf = CNF(from_string=dimacs_str)
    cubes = gen_cubes(cnf, min(2, cnf.nv))
    satisfaible_cubes = []

    # Prune unsatisfiable assumptions
    for cube in cubes:
        if check_sat(cnf, assumptions=cube):
            satisfaible_cubes.append(cube)

    cnf_tasks = []
    for cube in satisfaible_cubes:
        output_file = '/tmp/{}.cnf'.format(str(uuid.uuid1()))
        new_clauses = copy.deepcopy(clauses)
        # every list in the cube should be a unit clause?
        for lit in cube:
            new_clauses.append(str(lit))
        new_header = ["p cnf {0} {1}".format(cnf.nv, len(clauses) + len(cube))]
        write_dimacs_to_file(new_header, new_clauses, output_file)
        # TODO: write to different cnf files
        cnf_tasks.append(output_file)

    results = []
    pool = multiprocessing.Pool(processes=cpu_count())  # process pool
    for i in range(len(cnf_tasks)):
        result = pool.apply_async(call_sharp_sat,
                                  (cnf_tasks[i],))
        results.append(result)

    raw_solutions = []
    for i in range(len(cnf_tasks)):
        result = results[i].get()
        raw_solutions.append(int(result))

    print("results: ", raw_solutions)
    return sum(raw_solutions)
