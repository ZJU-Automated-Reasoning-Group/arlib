import argparse
import multiprocessing
import time

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

g_process_queue = []
sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
                        'g4', 'lgl', 'mcb', 'mpl', 'mg3',
                        'mc', 'm22', 'msh']


def solve_sat(solver_name: str, cnf: CNF, result_queue):
    aux = Solver(name=solver_name, bootstrap_with=cnf)
    ret = ""
    if aux.solve():
        ret = "sat"
    else:
        ret = "unsat"
    result_queue.put(ret)


def translate_smt_to_cnf(file_name: str):
    fml_vec = z3.parse_smt2_file(file_name)
    if len(fml_vec) == 1:
        fml = fml_vec[0]
    else:
        fml = z3.And(fml_vec)
    qfbv_preamble = z3.AndThen(z3.With('simplify', flat_and_or=False),
                               z3.With('propagate-values', flat_and_or=False),
                               z3.Tactic('elim-uncnstr'),
                               z3.With('solve-eqs', solve_eqs_max_occs=2),
                               z3.Tactic('reduce-bv-size'),
                               z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                                       local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
                               z3.With('simplify', hoist_mul=False, som=False, flat_and_or=False),
                               'max-bv-sharing',
                               'ackermannize_bv',
                               'bit-blast',
                               z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
                               z3.With('solve-eqs', solve_eqs_max_occs=2),
                               'aig',
                               'tseitin-cnf',
                               # z3.Tactic('sat')
                               )

    qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)
    after_simp = qfbv_tactic(fml).as_expr()
    if z3.is_false(after_simp):
        return "false", CNF()
    elif z3.is_true(after_simp):
        return "true", CNF()
    g = z3.Goal()
    g.add(after_simp)
    return "not sure", CNF(from_string=g.dimacs())


def benchmark_z3(file_name: str):
    start = time.process_time()
    fml_vec = z3.parse_smt2_file(file_name)
    if len(fml_vec) == 1:
        fml = fml_vec[0]
    else:
        fml = z3.And(fml_vec)
    s = z3.Solver()
    s.add(fml)
    end = time.process_time()
    print("z3       : ", s.check(), end - start)


def main():
    # global process_queue
    parser = argparse.ArgumentParser(description="Solve given formula.")
    parser.add_argument("formula", metavar="formula_file", type=str,
                        help="path to .smt2 formula file")
    parser.add_argument('--workers', dest='workers', default=8, type=int, help="num threads")
    parser.add_argument('--timeout', dest='timeout', default=60, type=int, help="timeout")

    args = parser.parse_args()

    formula_file = args.formula
    benchmark_z3(formula_file)
    start = time.process_time()
    finish, cnf = translate_smt_to_cnf(formula_file)
    if finish == "true":
        result = "sat"
    elif finish == "false":
        result = "unsat"
    else:
        result_queue = multiprocessing.Queue()
        n_workers = min(multiprocessing.cpu_count(), int(args.workers))
        sat_solver_to_use = sat_solvers_in_pysat[0:n_workers]
        with multiprocessing.Manager() as manager:
            result_queue = multiprocessing.Queue()
            for solver in sat_solver_to_use:
                g_process_queue.append(multiprocessing.Process(target=solve_sat,
                                                               args=(solver,
                                                                     cnf,
                                                                     result_queue
                                                                     )))

        for process in g_process_queue:
            process.start()
        try:
            result = result_queue.get(timeout=int(args.timeout))
        except multiprocessing.queues.Empty:
            result = "unknown"
        for p in g_process_queue:
            p.terminate()
    end = time.process_time()
    print("portfolio: ", result, end - start)


if __name__ == "__main__":
    main()
