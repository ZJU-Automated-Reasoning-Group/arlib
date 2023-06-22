#!/usr/bin/env python3
from pysat.solvers import Solver
from pysat.formula import CNF
import z3
import argparse
import multiprocessing
import json
import sys
import os
g_process_queue = []
sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
               'g4', 'lgl', 'mcb', 'mpl', 'mg3',
               'mc', 'm22']

# TODO : add about 4~5 kinds of z3 tactics
preambles = [
    z3.AndThen(z3.With('simplify', flat_and_or=False),
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
        ),
    z3.AndThen(z3.With('simplify', flat_and_or=False),
            z3.With('propagate-values', flat_and_or=False),
            z3.With('solve-eqs', solve_eqs_max_occs=2),
            z3.Tactic('elim-uncnstr'),
            z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                    local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
            z3.Tactic('max-bv-sharing'),
            z3.Tactic('bit-blast'),
            z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
            'aig',
            'tseitin-cnf',
    )
]

def solve_sat(solver_name : str, cnf : CNF, result_queue):
    aux = Solver(name=solver_name, bootstrap_with=cnf)
    print(solver_name, aux.solve())
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
    elif z3.is_false(after_simp):
        result_queue.put("unsat")
    # solve with sat solver
    else:
        g = z3.Goal()
        g.add(after_simp)
        cnf = CNF(from_string=g.dimacs())
        with multiprocessing.Manager() as manager:
            queue = multiprocessing.Queue()
            sub_processes = []
            for solver in sat_solvers_in_pysat:
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

def solve_with_z3(fml, result_queue):
    solver = z3.Solver()
    solver.add(fml)
    result_queue.put(solver.check().__str__())

def solve(file_name) -> str:
    fml_vec = z3.parse_smt2_file(file_name)
    if len(fml_vec) == 1:
        fml = fml_vec[0]
    else:
        fml = z3.And(fml_vec)

    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()
        for preamble in preambles:
            g_process_queue.append(multiprocessing.Process(target=preprocess_and_solve_sat,
                                                        args=(fml,
                                                                preamble,
                                                                result_queue
                                                                )))
            # use z3 as a single process
        g_process_queue.append(multiprocessing.Process(target=solve_with_z3,
                                                        args=(fml, result_queue)))

    for process in g_process_queue:
        process.start()
    try:
        result = result_queue.get()
    except multiprocessing.queues.Empty:
        result = "unknown"
    # Terminate all
    for p in g_process_queue:
        p.terminate()
    return result

def main():
    request_directory = sys.argv[1]
    input = os.path.join(request_directory, "input.json")
    with open(input) as f:
        input_json = json.loads(f.read())
    formula_file = input_json.get("formula_file")
    result = solve(formula_file)
    if result == "sat":
        res = 10
    elif result == "unsat":
        res = 20
    else:
        res = 0
    print(result, res)
    solver_output = {
        "return_code": res,
        "artifacts": {
            "stdout_path": os.path.join(request_directory, "stdout.log"),
            "stderr_path": os.path.join(request_directory, "stderr.log")
        }
    }
    
    with open(os.path.join(request_directory, "solver_out.json"), "w+") as f:
        f.write(json.dumps(solver_output))

if __name__ == "__main__":
    main()