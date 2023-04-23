from pysat.solvers import Solver
from pysat.formula import CNF
import z3
import argparse
import multiprocessing
import time
g_process_queue = []
sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
               'g4', 'lgl', 'mcb', 'mpl', 'mg3',
               'mc', 'm22', 'msh']

# todo : add about 4~5 kinds of z3 tictacs
preambles = []

def solve_sat(solver_name : str, cnf : CNF, result_queue):
    aux = Solver(name=solver_name, bootstrap_with=cnf)
    ret = ""
    if aux.solve():
        ret = "sat"
    else:
        ret = "unsat"
    result_queue.put(ret)

def preprocess_and_solve_sat(fml, qfbv_preamble, result_queue) :
    qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)
    after_simp = qfbv_tactic(fml).as_expr()
    # if z3 solve the problem directly
    if z3.is_true(after_simp):
        result_queue.put("sat")
    # solve with sat solver
    else:
        g = z3.Goal()
        g.add(after_simp)
        cnf = CNF(from_string=g.dimacs())
        with multiprocessing.Manager() as manager:
            queue = multiprocessing.Queue()
            sub_processes = []
            for solver in sat_solvers_in_pysat[0:9]:
                sub_processes.append(multiprocessing.Process(target=solve_sat,
                                                            args=(solver,
                                                                    cnf,
                                                                    queue
                                                                    )))
            for process in sub_processes:
                process.start()
            try:
                result = queue.get()
            except multiprocessing.queues.Empty:
                result = "unknown"
        result_queue.put(result)

def main():
    parser = argparse.ArgumentParser(description="Solve given formula.")
    parser.add_argument("formula", metavar="formula_file", type=str,
                        help="path to .smt2 formula file")
    parser.add_argument('--workers', dest='workers', default=8, type=int, help="num threads")
    parser.add_argument('--timeout', dest='timeout', default=600, type=int, help="timeout")

    args = parser.parse_args()
    fml_vec = z3.parse_smt2_file(args.formula)
    if len(fml_vec) == 1:
        fml = fml_vec[0]
    else:
        fml = z3.And(fml_vec)

    start = time.process_time()
    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()
        for preamble in preambles:
            g_process_queue.append(multiprocessing.Process(target=preprocess_and_solve_sat,
                                                        args=(fml,
                                                                preamble,
                                                                result_queue
                                                                )))

    for process in g_process_queue:
        process.start()
    try:
        result = result_queue.get(timeout=int(args.timeout))
    except multiprocessing.queues.Empty:
        result = "unknown"
    # Terminate all
    for p in g_process_queue:
        p.terminate()
    end = time.process_time()
    print("portfolio:", result, end - start)

if __name__ == "__main__":
    main()