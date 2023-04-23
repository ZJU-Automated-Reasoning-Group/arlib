"""
This file will use fuzzsat.py, smtfuzz.py, and fuzzqbf.py to generate more complex formula

"""

import logging
import random
import subprocess
from pathlib import Path
from threading import Timer
from typing import List

CNF_GENERATOR = str(Path(__file__).parent) + "/fuzzsat.py"
QBF_GENERATOR = str(Path(__file__).parent) + "/fuzzqbf.py"
SMT_GENERATOR = str(Path(__file__).parent) + "/smtfuzz.py"


def terminate(process, is_timeout: List):
    """
    Terminates a process and sets the timeout flag to True.
    Parameters:
    -----------
    process : subprocess.Popen
        The process to be terminated.
    is_timeout : List
        A list containing a single boolean item.
        If the process exceeds the timeout limit, the boolean item will be
        set to True.

    Returns:
    --------
    None
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print("error for interrupting")
            print(ex)


def gen_cnf_numeric_clauses() -> List[List[int]]:
    """
    Generate a CNF formula in the form of numeric clauses
    FIXME: fuzzsat generates a 0 at the end of each line
      but pysat does not like 0
    """
    print(CNF_GENERATOR)
    cmd = ['python3', CNF_GENERATOR,
           '-i', str(random.randint(1, 10)),
           '-I', str(random.randint(11, 50)),
           '-p', str(random.randint(2, 10)),
           '-P', str(random.randint(11, 30)),
           '-l', str(random.randint(2, 10)),
           '-L', str(random.randint(11, 30))]

    logging.debug("Enter constraint generation")
    logging.debug(cmd)
    # Create a subprocess and assign it to p_gene
    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Set is_timeout_gene to False
    is_timeout_gene = [False]
    # Create a timer that will terminate the process after 15 seconds
    timer_gene = Timer(15, terminate, args=[p_gene, is_timeout_gene])
    timer_gene.start()
    # Read the output of the process
    out_gene = p_gene.stdout.readlines()
    # Decode the output from bytes to string
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    # Close the output stream
    p_gene.stdout.close()
    # Cancel the timer
    timer_gene.cancel()
    # If the process did not time out, try to parse the output
    if not is_timeout_gene[0]:
        result = []
        try:
            # Split the output by line
            for line in out_gene.split("\n"):
                data = line.split(" ")
                # If the first element is empty and there are more than one elements, append the integers to the result
                if data[0] == '' and len(data) > 1:
                    result.append([int(dd) for dd in data[1:-1]])
            # If the process is still running, terminate it
            if p_gene.poll() is None:
                p_gene.terminate()
            return result
        except Exception as ex:
            print(ex)
    # If the process is still running, terminate it
    if p_gene.poll() is None:
        p_gene.terminate()
    # Return an empty list
    return []


def gene_smt2string(logic="QF_BV", incremental=False) -> str:
    """
    Generate an SMT-LIB2 string
    :param logic:
    :param incremental:
    :return: a string
    """
    cnfratio = random.randint(2, 10)
    cntsize = random.randint(5, 20)
    # strategy = random.choice(strategies)
    if incremental:
        strategy = random.choice(['cnf', 'ncnf', 'bool'])
    else:
        strategy = 'noinc'

    # 'CNFexp'

    cmd = ['python3', SMT_GENERATOR,
           '--strategy', strategy,
           '--cnfratio', str(cnfratio),
           '--cntsize', str(cntsize),
           '--disable', 'option_fuzzing',
           '--difftest', '1',
           '--logic', logic]

    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout_gene = [False]
    timer_gene = Timer(6, terminate, args=[p_gene, is_timeout_gene])
    timer_gene.start()
    out_gene = p_gene.stdout.readlines()
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    p_gene.stdout.close()  # close?
    timer_gene.cancel()
    if p_gene.poll() is None:
        p_gene.terminate()  # need this?
    if is_timeout_gene[0]:
        return False
    return out_gene


def generate_from_grammar_as_str(logic="QF_BV", incremental=False):
    cnfratio = random.randint(2, 10)
    cntsize = random.randint(5, 20)

    # strategy = random.choice(strategies)
    if incremental:
        strategy = random.choice(['CNFexp', 'cnf', 'ncnf', 'bool'])
    else:
        strategy = 'noinc'

    cmd = ['python3', SMT_GENERATOR,
           '--strategy', strategy,
           '--cnfratio', str(cnfratio),
           '--cntsize', str(cntsize),
           '--disable', 'option_fuzzing',
           '--difftest', '1',
           '--logic', logic]

    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout_gene = [False]
    timer_gene = Timer(15, terminate, args=[p_gene, is_timeout_gene])
    timer_gene.start()
    out_gene = p_gene.stdout.readlines()
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    p_gene.stdout.close()  # close?
    timer_gene.cancel()
    if p_gene.poll() is None:
        p_gene.terminate()  # need this?
    if is_timeout_gene[0]:
        return False
    return out_gene


if __name__ == '__main__':
    print(gene_smt2string())
