# coding: utf-8
"""
Process-based Parallel CDCL(T)-style SMT Solving
"""

import logging
import multiprocessing
from multiprocessing import cpu_count, Queue, Process
from multiprocessing import Manager
from ctypes import c_char_p
from typing import List, Optional, Tuple, Set

from arlib.smt.pcdclt import SMTPreprocessor4Process, BooleanFormulaManager
from arlib.bool import PySATSolver, simplify_numeric_clauses
from arlib.smt.pcdclt.theory import SMTLibTheorySolver
from arlib.utils import SolverResult, SExprParser
from arlib.smt.pcdclt.exceptions import TheorySolverSuccess, PySMTSolverError
from arlib.utils.exceptions import SMTLIBSolverError
from arlib.config import m_smt_solver_bin

logger = logging.getLogger(__name__)

"""

# Some options to be configured (assuming use Z3 for now
"""
M_SIMPLIFY_BLOCKING_CLAUSES = True
DEFAULT_SAMPLE_SIZE = 10


# End of options


def theory_worker(worker_id, init_theory_fml, task_queue, result_queue, bin_solver):
    """
    Long-running theory worker process that handles multiple theory checks.
    
    :param worker_id: ID of this worker
    :param init_theory_fml: The initial theory formula (constant throughout solving)
    :param task_queue: Queue to receive new tasks (assumptions to check)
    :param result_queue: Queue to send back results
    :param bin_solver: Path to binary solver executable
    """
    logger.debug(f"Theory worker {worker_id} started with solver: {bin_solver}")
    
    # Initialize the theory solver once
    theory_solver = SMTLibTheorySolver(bin_solver)
    theory_solver.add(init_theory_fml.value)
    
    while True:
        # Get next task from queue
        task_id, assumptions = task_queue.get()
        
        # Special signal to terminate
        if task_id == -1:
            logger.debug(f"Theory worker {worker_id} shutting down")
            break
            
        try:
            # Check the current assumptions
            logger.debug(f"Worker {worker_id} checking assumptions")
            if SolverResult.UNSAT == theory_solver.check_sat_assuming(assumptions):
                unsat_core = theory_solver.get_unsat_core()
                result_queue.put((task_id, unsat_core))
            else:
                # Signal that this model is theory-consistent (SAT)
                result_queue.put((task_id, ""))
        except Exception as e:
            # Handle any errors
            logger.error(f"Worker {worker_id} encountered error: {str(e)}")
            result_queue.put((task_id, f"ERROR:{str(e)}"))


def theory_solve_with_workers(all_assumptions: List[str], task_queue, result_queue, num_workers):
    """
    Submit theory solving tasks to worker processes and collect results.
    
    :param all_assumptions: List of sets of assumptions to check
    :param task_queue: Queue to send tasks to workers
    :param result_queue: Queue to receive results from workers
    :param num_workers: Number of worker processes
    :return: List of unsat cores from theory solvers
    """
    # Submit all tasks to the queue
    for i, assumptions in enumerate(all_assumptions):
        task_queue.put((i, assumptions))
    
    # Collect all results
    raw_unsat_cores = []
    for _ in range(len(all_assumptions)):
        task_id, result = result_queue.get()
        
        if result.startswith("ERROR:"):
            logger.error(f"Theory solving error: {result}")
            continue
            
        if result == "":  # empty core indicates SAT
            # If any model is theory-consistent, we're done
            return []
            
        raw_unsat_cores.append(result)
    
    return raw_unsat_cores


def parse_raw_unsat_core(core: str, bool_manager: BooleanFormulaManager) -> List[int]:
    """
     Given an unsat core in string, build a blocking numerical clause from the core
    :param core: The unsat core built a theory solver
    :param bool_manager: The manger for tracking the information of Boolean abstraciton
         (e.g., the mapping between the name and the numerical ID)
    :return: The blocking clauses built from the unsat core
    """
    parsed_core = SExprParser.parse_sexpr_string(core)
    print(parsed_core)
    assert len(parsed_core) >= 1
    # Let the parsed_core be ['p@4', 'p@7', ['not', 'p@6']]
    blocking_clauses_core = []  # map the core to a numerical clause
    # the blocking clauses should be (not (and 4 7 -6)), i.e., [-4, -7, 6]
    for ele in parsed_core:
        if isinstance(ele, list):
            blocking_clauses_core.append(bool_manager.vars2num[ele[1]])
        else:
            blocking_clauses_core.append(-bool_manager.vars2num[ele])
    return blocking_clauses_core


def process_pysat_models(bool_models: List[List[int]],
                         bool_manager: BooleanFormulaManager) -> List[str]:
    """
    Given a set of Boolean models, built the assumptions (to be checked by the theory solvers)
    :param bool_models: The set of Boolean models
    :param bool_manager: The manger for tracking the information of Boolean abstraciton
         (e.g., the mapping between the name and the numerical ID)
    :return: The set of assumptions
    """
    all_assumptions = []
    for model in bool_models:
        assumptions = []
        for val in model:
            if val > 0:
                assumptions.append(bool_manager.num2vars[val])
            else:
                assumptions.append("(not {})".format(bool_manager.num2vars[abs(val)]))
        all_assumptions.append(assumptions)

    return all_assumptions


def parallel_cdclt_process(smt2string: str, logic: str, num_samples_per_round=10) -> SolverResult:
    """
    The main function of the parallel CDCL(T) SMT solving enigne
    :param smt2string: The formula to be solved
    :param logic: The logic type
    :return: The satisfiability result
    """
    preprocessor = SMTPreprocessor4Process()
    bool_manager, th_manager = preprocessor.from_smt2_string(smt2string)

    logger.debug("Finish preprocessing")

    if preprocessor.status != SolverResult.UNKNOWN:
        logger.debug("Solved by the preprocessor")
        return preprocessor.status

    # Use number of CPUs as worker count
    num_workers = cpu_count()
    
    # Initialize task and result queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Initialize boolean solver
    bool_solver = PySATSolver()
    bool_solver.add_clauses(bool_manager.numeric_clauses)

    # Prepare theory formula (remains constant throughout solving)
    init_theory_fml_str = " (set-logic {}) ".format(logic) + \
                          " (set-option :produce-unsat-cores true) " + \
                          " ".join(th_manager.smt2_signature) + \
                          "(assert {})".format(th_manager.smt2_init_cnt)

    # Create a shared version of the theory formula
    manager = Manager()
    init_theory_fml_str_shared = manager.Value(c_char_p, init_theory_fml_str)

    # Start worker processes
    workers = []
    for i in range(num_workers):
        p = Process(target=theory_worker, 
                   args=(i, init_theory_fml_str_shared, task_queue, result_queue, m_smt_solver_bin))
        p.daemon = True
        p.start()
        workers.append(p)
    
    logger.debug(f"Started {num_workers} theory worker processes")

    sample_number = num_samples_per_round
    result = SolverResult.UNKNOWN
    
    try:
        while True:
            is_sat = bool_solver.check_sat()
            if not is_sat:
                result = SolverResult.UNSAT
                break
                
            logger.debug("Boolean abstraction is satisfiable")
            bool_models = bool_solver.sample_models(to_enum=sample_number)
            logger.debug(f"Finish sampling {len(bool_models)} Boolean models; Start checking T-consistency!")

            all_assumptions = process_pysat_models(bool_models, bool_manager)
            raw_unsat_cores = theory_solve_with_workers(all_assumptions, task_queue, result_queue, num_workers)

            if len(raw_unsat_cores) == 0:
                result = SolverResult.SAT
                break

            logger.debug("Theory solvers finished; Raw unsat cores: {}".format(raw_unsat_cores))
            blocking_clauses = []
            for core in raw_unsat_cores:
                blocking_clause = parse_raw_unsat_core(core, bool_manager)
                blocking_clauses.append(blocking_clause)

            logger.debug("Blocking clauses from theory solver: {}".format(blocking_clauses))

            if M_SIMPLIFY_BLOCKING_CLAUSES:
                blocking_clauses = simplify_numeric_clauses(blocking_clauses)
                logger.debug("After simplifications: {}".format(blocking_clauses))

            for cls in blocking_clauses:
                bool_solver.add_clause(cls)

    except Exception as e:
        logger.error(f"Error in main solving loop: {str(e)}")
        result = SolverResult.UNKNOWN
    finally:
        # Signal all workers to terminate
        for _ in range(num_workers):
            task_queue.put((-1, None))  # -1 is the signal to terminate
        
        # Wait for workers to terminate (with timeout)
        for p in workers:
            p.join(timeout=1.0)
            
        # Force terminate any remaining workers
        for p in workers:
            if p.is_alive():
                p.terminate()

    return result
