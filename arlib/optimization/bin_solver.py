"""
For calling bin solvers
"""
import os
import subprocess
import logging
import uuid
from typing import List
from threading import Timer

import z3

from arlib.global_params import global_config


logger = logging.getLogger(__name__)

g_bin_solver_timeout = 100

def terminate(process, is_timeout: List):
    """
    Terminate a process and set the timeout flag.
    Args:
        process (subprocess.Popen): The process to be terminated.
        is_timeout (List[bool]): A list containing a single boolean item.
                                 It will be set to True if the process
                                 exceeds the timeout limit.
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
            logger.debug("Process terminated due to timeout.")
        except Exception as ex:
            print("error for interrupting")
            logger.error("Error interrupting process: %s", ex)


def get_smt_solver_command(solver_name: str, tmp_filename: str) -> List[str]:
    """
    Get the command to run the specified solver.
    Args:
        solver_name (str): The name of the solver.
        tmp_filename (str): The temporary file name containing the SMT problem.

    Returns:
        List[str]: The command to execute the solver.
    """
    solvers = {
        "z3": [global_config.z3_exec, tmp_filename],
        "cvc5": [global_config.cvc5_exec, "-q", "--produce-models", tmp_filename],
        "btor": [global_config.btor_exec, tmp_filename],
        "yices": [global_config.yices_exec, tmp_filename],
        "mathsat": [global_config.math_exec, tmp_filename],
        "bitwuzla": [global_config.bitwuzla_exec, tmp_filename],
        "q3b": [global_config.q3b_exec, tmp_filename],
    }
    return solvers.get(solver_name, [global_config.z3_exec, tmp_filename])


def get_maxsat_solver_command(solver_name: str, tmp_filename: str) -> List[str]:
    """
    Get the command to run the specified solver.
    Args:
        solver_name (str): The name of the solver.
        tmp_filename (str): The temporary file name containing the SMT problem.

    Returns:
        List[str]: The command to execute the solver.
    """
    solvers = {
        "z3": [global_config.z3_exec, tmp_filename],
    }
    return solvers.get(solver_name, [global_config.z3_exec, tmp_filename])


def solve_with_bin_smt(logic: str, qfml: z3.ExprRef, obj_name: str, solver_name: str):
    """
    Call binary SMT solvers to solve quantified SMT problems.

    Args:
        logic (str): The logic to be used.
        qfml (z3.ExprRef): The formula to be solved.
        obj_name: The name of the objective
        solver_name (str): The name of the solver to use.

    Returns:
        str: The result of the solver ('sat', 'unsat', or 'unknown').
    """
    logger.debug("Solving QSMT via {}".format(solver_name))
    fml_str = "(set-option :produce-models true)\n"
    fml_str += "(set-logic {})\n".format(logic)
    s = z3.Solver()
    s.add(qfml)
    fml_str += s.to_smt2()
    fml_str += "(get-value ({}))\n".format(obj_name)
    # print(fml_str)
    tmp_filename = "/tmp/{}_temp.smt2".format(str(uuid.uuid1()))
    try:
        tmp = open(tmp_filename, "w")
        tmp.write(fml_str)
        tmp.close()
        cmd = get_smt_solver_command(solver_name, tmp_filename)
        # print(cmd)
        logger.debug("Command: %s", cmd)
        p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        is_timeout_gene = [False]
        timer_gene = Timer(g_bin_solver_timeout, terminate, args=[p_gene, is_timeout_gene])
        timer_gene.start()
        out_gene = p_gene.stdout.readlines()
        out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
        p_gene.stdout.close()  # close?
        timer_gene.cancel()

        if p_gene.poll() is None:
            p_gene.terminate()  # TODO: need this?

        # FIXME: parse the model returned by the SMT solver?
        if is_timeout_gene[0]:
            return "unknown"
        elif "unsat" in out_gene:
            return out_gene
            # return "unsat"
        elif "sat" in out_gene:
            return out_gene
            # return "sat"
        else:
            return "unknown"
    finally:
        # tmp.close()
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_maxsat(wcnf: str, solver_name: str):
    """
    Solve weighted MaxSAT via binary solvers (Maybe we can use pipe to send the
      instance to the solvers..?)
    """
    logger.debug("Solving QSMT via {}".format(solver_name))
    fml_str = ""
    tmp_filename = "/tmp/{}_temp.wcnf".format(str(uuid.uuid1()))
    try:
        tmp = open(tmp_filename, "w")
        tmp.write(fml_str)
        tmp.close()
        cmd = get_maxsat_solver_command(solver_name, tmp_filename)
        logger.debug("Command: %s", cmd)
        p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        is_timeout_gene = [False]
        timer_gene = Timer(g_bin_solver_timeout, terminate, args=[p_gene, is_timeout_gene])
        timer_gene.start()
        out_gene = p_gene.stdout.readlines()
        out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
        p_gene.stdout.close()  # close?
        timer_gene.cancel()

        if p_gene.poll() is None:
            p_gene.terminate()  # TODO: need this?

        # FIXME: parse the model returned by the SMT solver?
        if is_timeout_gene[0]:
            return "unknown"
        elif "xxxxx" in out_gene:
            return out_gene
        elif "xxxxx" in out_gene:
            return out_gene
        else:
            return "unknown"
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def demo_solver():
    cmd = [global_config.z3_exec, 'tmp.smt2']
    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout_gene = [False]
    timer_gene = Timer(g_bin_solver_timeout, terminate, args=[p_gene, is_timeout_gene])
    timer_gene.start()
    out_gene = p_gene.stdout.readlines()
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    p_gene.stdout.close()  # close?
    timer_gene.cancel()
    if p_gene.poll() is None:
        p_gene.terminate()  # TODO: need this?

    print(out_gene)


if __name__ == "__main__":
    demo_solver()
