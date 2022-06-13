# coding: utf-8
from typing import List
import random
import subprocess
from threading import Timer
import logging

from pdsmt.tests.formula_generator import FormulaGenerator
from z3 import *
from pdsmt.simple_cdclt import simple_cdclt, boolean_abstraction
from pdsmt.bool.pysat_solver import PySATSolver

logging.basicConfig(level=logging.INFO)


def terminate(process, is_timeout):
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as e:
            print("error for interrupting")
            print(e)


def gen_cnf_fml():
    """
    FIXME: fuzzsat generates a 0 at the end of each line
      but pysat does not like 0
    """
    generator = os.getcwd() + '/pdsmt/tests/fuzzsat.py'
    cmd = ['python3', generator]
    cmd.append('-i')
    cmd.append(str(random.randint(1, 10)))
    cmd.append('-I')
    cmd.append(str(random.randint(11, 50)))
    cmd.append('-p')
    cmd.append(str(random.randint(2, 10)))
    cmd.append('-P')
    cmd.append(str(random.randint(11, 30)))
    cmd.append('-l')
    cmd.append(str(random.randint(2, 10)))
    cmd.append('-L')
    cmd.append(str(random.randint(11, 30)))

    logging.debug("Enter constraint generation")
    logging.debug(cmd)
    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout_gene = [False]
    timer_gene = Timer(15, terminate, args=[p_gene, is_timeout_gene])
    timer_gene.start()
    out_gene = p_gene.stdout.readlines()
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    p_gene.stdout.close()  # close?
    timer_gene.cancel()
    if is_timeout_gene[0]:
        return []
    res = []
    try:
        for line in out_gene.split("\n"):
            data = line.split(" ")
            if data[0] == '' and len(data) > 1:
                res.append([int(d) for d in data[1:-1]])
    except Exception as ex:
        print(ex)
        # print(out_gene)
        return []
    return res


def test_sat():
    for _ in range(50):
        clauses = gen_cnf_fml()
        if len(clauses) == 0:
            continue
        # solver_name = random.choice(sat_solvers)
        s = PySATSolver()
        s.add_clauses(clauses)
        if s.check_sat():
            print("SAT")
            s.enumerate_models(10)
            # break


def test_smt():
    for _ in range(20):
        w, x, y, z = Ints("w x y z")
        fg = FormulaGenerator([w, x, y, z])
        smt2string = fg.generate_formula_as_str()
        res_z3sat = simple_cdclt(smt2string)
        print(res_z3sat)
        # res = boolean_abstraction(smt2string)
    print("Finished!")


if __name__ == "__main__":
    # test_sat()
    test_smt()
