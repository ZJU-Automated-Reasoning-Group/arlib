"""
For calling bin solvers
"""
import os
import subprocess
import logging
import uuid
from typing import List, Dict
from threading import Timer

import z3

from arlib.global_params import global_config

logger = logging.getLogger(__name__)
BIN_SOLVER_TIMEOUT = 100
# Result = Literal["sat", "unsat", "unknown"]

def terminate(process, is_timeout: List):
    """Terminate a process and set timeout flag."""
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
            logger.debug("Process terminated due to timeout.")
        except Exception as ex:
            logger.error("Error interrupting process: %s", ex)


def get_solver_command(solver_type: str, solver_name: str, tmp_filename: str) -> List[str]:
    """Get the command to run the specified solver."""
    solvers: Dict[str, Dict[str, List[str]]] = {
        "smt": {
            "z3": [global_config.z3_exec, tmp_filename],
            "cvc5": [global_config.cvc5_exec, "-q", "--produce-models", tmp_filename],
            "btor": [global_config.btor_exec, tmp_filename],
            "yices": [global_config.yices_exec, tmp_filename],
            "mathsat": [global_config.math_exec, tmp_filename],
            "bitwuzla": [global_config.bitwuzla_exec, tmp_filename],
            "q3b": [global_config.q3b_exec, tmp_filename],
        },
        "maxsat": {
            "z3": [global_config.z3_exec, tmp_filename],
        }
    }
    
    default = [global_config.z3_exec, tmp_filename]
    return solvers.get(solver_type, {}).get(solver_name, default)


def run_solver(cmd: List[str]) -> str:
    """Run solver command and handle timeout."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(BIN_SOLVER_TIMEOUT, terminate, args=[p, is_timeout])
    
    try:
        timer.start()
        out = p.stdout.readlines()
        out = ' '.join([line.decode('UTF-8') for line in out])
        
        if is_timeout[0]:
            return "unknown"
        elif "unsat" in out:
            return out
        elif "sat" in out:
            return out
        else:
            return "unknown"
    finally:
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        p.stdout.close()


def solve_with_bin_smt(logic: str, qfml: z3.ExprRef, obj_name: str, solver_name: str) -> str:
    """Call binary SMT solvers to solve quantified SMT problems."""
    logger.debug(f"Solving QSMT via {solver_name}")
    
    # Prepare SMT2 formula
    fml_str = "(set-option :produce-models true)\n"
    fml_str += f"(set-logic {logic})\n"
    s = z3.Solver()
    s.add(qfml)
    fml_str += s.to_smt2()
    fml_str += f"(get-value ({obj_name}))\n"
    
    # Create temporary file
    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.smt2"
    try:
        with open(tmp_filename, "w") as tmp:
            tmp.write(fml_str)
        
        cmd = get_solver_command("smt", solver_name, tmp_filename)
        logger.debug("Command: %s", cmd)
        return run_solver(cmd)
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_maxsat(wcnf: str, solver_name: str) -> str:
    """Solve weighted MaxSAT via binary solvers."""
    logger.debug(f"Solving MaxSAT via {solver_name}")
    
    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.wcnf"
    try:
        with open(tmp_filename, "w") as tmp:
            tmp.write(wcnf)
        
        cmd = get_solver_command("maxsat", solver_name, tmp_filename)
        logger.debug("Command: %s", cmd)
        return run_solver(cmd)
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def demo_solver():
    """Demo function to test solver functionality."""
    cmd = [global_config.z3_exec, 'tmp.smt2']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(BIN_SOLVER_TIMEOUT, terminate, args=[p, is_timeout])
    
    try:
        timer.start()
        out = p.stdout.readlines()
        out = ' '.join([line.decode('UTF-8') for line in out])
        print(out)
    finally:
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        p.stdout.close()


if __name__ == "__main__":
    demo_solver()
