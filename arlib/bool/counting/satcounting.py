"""
Counting solutions of a CNF formula
"""
import subprocess
from threading import Timer
from typing import List
import itertools
import logging
import os
import subprocess
from threading import Timer
from timeit import default_timer as counting_timer
# import multiprocessing
from typing import List

import z3

from arlib.utils.z3_expr_utils import get_variables
# import random
from arlib.bv.mapped_blast import translate_smt2formula_to_cnf_file

sharp_sat_bin = "???"


def terminate(process, is_timeout: List):
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            print(ex)


def clear_tmp_cnf_files():
    if os.path.isfile('/tmp/out.cnf'):
        os.remove('/tmp/out.cnf')


class SATModelCounter:
    """Model counter and SAT formula
    """
    def __init__(self):
        self.strategy = "sharpSAT"
        self.timeout = 30

    def smt2cnf(self):
        return

    def count_models_by_enumeration(self, smtfml: z3.ExprRef):
        """Try every assignment
        TODO: we do not need to solve, if constructing a model object and trying the eval function
        """
        tac = z3.Then('simplify', 'bit-blast')
        # TODO: tac seems to lead to inconsistent model counts (need to use mappedblast?...)
        bool_fml = tac(smtfml).as_expr()
        bool_vars = get_variables(bool_fml)
        time_start = counting_timer()
        solutions = 0
        solver = z3.Solver()
        solver.add(bool_fml)
        for assignment in itertools.product(*[(x, z3.Not(x)) for x in bool_vars]):  # all combinations
            if solver.check(assignment) == z3.sat:  # conditional check (does not add assignment permanently)
                solutions += 1
        print("Time:", counting_timer() - time_start)
        return solutions

    def count_models_by_knowledge_compilation(self, smtfml: z3.ExprRef):
        return

    def count_models_by_sharp_sat(self, smtfml: z3.ExprRef):
        solutions = 0
        time_start = counting_timer()
        clear_tmp_cnf_files()
        outputifle = '/tmp/out.cnf'
        # translate_smt2file_to_cnffile(self.smt2file, outputifle)
        # TODO: fml is a bit-vector formula, but not self.formula (which is a SAT formula). This is a bit strange
        translate_smt2formula_to_cnf_file(smtfml, outputifle)
        cmd = [sharp_sat_bin, outputifle]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        is_timeout = [False]
        timer = Timer(self.timeout, terminate, args=[p, is_timeout])
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
                if decode_line.startswith("# solu"):
                    find_sol_line = True
        except Exception as ex:
            print(ex)
            print("exception when running sharpSAT, will return false")
            clear_tmp_cnf_files()
        if is_timeout[0]: logging.debug("sharpSAT timeout")  # should we put it in the above try scope?
        p.stdout.close()  # close?
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        # process time is not current (it seems to miss the time spent on sharpSAT
        # print("Time:", counting_timer() - time_start)
        clear_tmp_cnf_files()
        return solutions

    def cube_and_conquer_sharp_sat(self, smtfml):
        """
        1. Generate a set of disjont cubes such that they can be extended to be models of the formula
            C1: a, b
            C2: Not(a), b
            C3: a, Not b
            C4: Not(a), Not(b)
        2. Count the models subject to each cube in parallel
        """
        solutions = 0
        # time_start = counting_timer()
        clear_tmp_cnf_files()
        outputifle = '/tmp/out.cnf'
        # translate_smt2file_to_cnffile(self.smt2file, outputifle)
        # TODO: fml is a bit-vector formula, but not self.formula (which is a SAT formula). This is a bit strange
        translate_smt2formula_to_cnf_file(smtfml, outputifle)
        # TODO: parse, partition, and count in parallel?
        return solutions
