# coding: utf-8
import logging
import random
import subprocess
from threading import Timer

from z3 import *

from pdsmt.bool.pysat_solver import PySATSolver
from pdsmt.parallel_cdclt import parallel_cdclt
from pdsmt.tests.formula_generator import FormulaGenerator


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
    cmd = ['python3', generator, '-i', str(random.randint(1, 10)), '-I', str(random.randint(11, 50)), '-p',
           str(random.randint(2, 10)), '-P', str(random.randint(11, 30)), '-l', str(random.randint(2, 10)), '-L',
           str(random.randint(11, 30))]

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
            models = s.sample_models(10)
            reduced = s.reduce_models(models)
            print(models)
            print(reduced)
            # break


def test_smt():
    for _ in range(33):
        w, x, y, z = Reals("w x y z")
        fg = FormulaGenerator([w, x, y, z])
        smt2string = fg.generate_formula_as_str()
        # res = simple_cdclt(smt2string)
        res = parallel_cdclt(smt2string, logic="ALL")
        print(res)
        # res = boolean_abstraction(smt2string)
    print("Finished!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sat()
    # test_smt()
