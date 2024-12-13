# coding: utf-8
"""
Flattening-based QF_BV solver
"""
import z3
from pysat.formula import CNF
from pysat.solvers import Solver
import multiprocessing
from threading import Timer

import sys
from pathlib import Path

project_root_dir = str(Path(__file__).parent.parent)
sys.path.append(project_root_dir)

from arlib.utils import SolverResult
from arlib.bool.features.sat_instance import *
import os

from concurrent.futures import *

"""
    cadical103  = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    cadical153  = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    gluecard3   = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    gluecard4   = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    glucose3    = ('g3', 'g30', 'glucose3', 'glucose30')
    glucose4    = ('g4', 'g41', 'glucose4', 'glucose41')
    lingeling   = ('lgl', 'lingeling')
    maplechrono = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    maplecm     = ('mcm', 'maplecm')
    maplesat    = ('mpl', 'maple', 'maplesat')
    mergesat3   = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    minicard    = ('mc', 'mcard', 'minicard')
    minisat22   = ('m22', 'msat22', 'minisat22')
    minisatgh   = ('mgh', 'msat-gh', 'minisat-gh')
"""
sat_solvers_in_pysat = [ 'gc3', 'gc4', 'g3',
                        'g4',  'mcb', 'mpl', 'mg3',
                        'mc', 'm22', 'mgh']

qfbv_preamble = z3.AndThen(z3.With('simplify', flat_and_or=False),
                           z3.With('propagate-values', flat_and_or=False),
                           z3.Tactic('elim-uncnstr'),
                           z3.With('solve-eqs', solve_eqs_max_occs=2),
                           z3.Tactic('reduce-bv-size'),
                           z3.With('simplify', som=True, pull_cheap_ite=True, push_ite_bv=False, local_ctx=True,
                                   local_ctx_limit=10000000, flat=True, hoist_mul=False, flat_and_or=False),
                           # Z3 can solve a couple of extra benchmarks by using hoist_mul but the timeout in SMT-COMP is too small.
                           # Moreover, it impacted negatively some easy benchmarks. We should decide later, if we keep it or not.
                           # With('simplify', hoist_mul=False, som=False, flat_and_or=False),
                           z3.Tactic('max-bv-sharing'),
                           z3.Tactic('ackermannize_bv'),
                           z3.Tactic('bit-blast'),
                           z3.With('simplify', local_ctx=True, flat=False, flat_and_or=False),
                           # With('solve-eqs', local_ctx=True, flat=False, flat_and_or=False),
                           z3.Tactic('tseitin-cnf'),
                           # z3.Tactic('sat')
                           )
qfbv_tactic = z3.With(qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True)

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)

    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# sat_solver_name = "minisat22"
# max solving time is limited to 2500s
MAX_WAIT = 2500


def solve_smt_file(self, filepath: str):
    fml_vec = z3.parse_smt2_file(filepath)
    print(fml_vec)
    return self.check_sat(z3.And(fml_vec))

# 还没跑完的
# dataset_to_build = ["uum", "VS3", "2018-Goel-hwbench", "bench_ab", "bmc-bv","20190311-bv-term-small-rw-Noetzli",
#                     "bmc-bv-svcomp14","brummayerbiere","brummayerbiere2","brummayerbiere3",
#                      "Sage2", "fft", "pspace" ,
#                     ]

# dataset_to_build = ["calypto" , "float","galois", "mcm"]

dataset_hard_to_solve = ["uum" ]

dataset_to_build = ["log-slicing" , "gulwani-pldi08" , "Booth", "VS3", "2018-Goel-hwbench" , "bench_ab" ,
                    "20190311-bv-term-small-rw-Noetzli" , "bmc-bv-svcomp14" , "brummayerbiere" , "brummayerbiere2",
                    "brummayerbiere3" , "fft" , "pspace" ,
                    ]

# dataset_to_build = ["Sage2_1" , "Sage2_2" , "Sage2_3" , "Sage2_4" , "Sage2_5" , "Sage2_6" , "Sage2_7"
#                     , "Sage2_8"  , "Sage2_9" ,  "Sage2_10",
#                     ]


dataset_has_been_built = ["2017-BuchwaldFried","RWS","2019-Mann","bmc-bv-svcomp14","lfsr","2018-Mann",
                          "simple_processor" , "uclid_contrib_smtcomp09" , "2019A" , "ecc" ,"2018E", "stp_samples",
                          "rubik" ,
                          ]


def interrupt(s):
    s.interrupt()

def check_sat(solver_name : str , filepath: str) -> (int, int):
    # parsing cnf files
    pos = CNF(from_file=filepath)
    aux = Solver(name=solver_name, bootstrap_with=pos,use_timer=True)
    timer = Timer(MAX_WAIT, interrupt, [aux])
    timer.start()
    is_solved = aux.solve_limited(expect_interrupt=True)
    if is_solved is None :
        raise TimeoutException
    solving_time = aux.time()
    res = (SolverResult.SAT, solving_time) if is_solved else (SolverResult.UNSAT, solving_time)
    # print(f"Solving {filepath} via  {solver_name} takes {solving_time} , {res[0]}")
    return res


def building_thread(files, dir_path):
    data_file = open(dir_path + "/text.txt", "a")
    log_file = open(dir_path + "/log.txt", "a")
    for file in files:
        if not file.endswith(".cnf") : continue
        file_path = os.path.join(dir_path, file)
        sat_inst = get_base_features(file_path)
        for k, v in sat_inst.features_dict.items():
            data_file.write(f"{v},")
        min_time_solver_taken = float('inf')
        min_time_solver_idx = -1
        for idx, sat_solver in enumerate(sat_solvers_in_pysat):
            try:
                (res, solving_time) = check_sat(solver_name=sat_solver,filepath=file_path)
                log_file.write(f"Solving {file_path} via  {sat_solver} takes {solving_time}\n")
                log_file.flush()
                if solving_time < min_time_solver_taken:
                    min_time_solver_idx = idx
                    min_time_solver_taken = solving_time
            except TimeoutException:
                log_file.write(f"Solving {file_path} via  {sat_solver} out of time\n")
                log_file.flush()
        data_file.write(f"{min_time_solver_idx},\n")
        data_file.flush()




if __name__ == "__main__":
    from pathlib import Path
    import sys

    project_root_dir = str(Path(__file__).parent.parent)
    sys.path.append(project_root_dir)
    # Get feature names
    # sat_inst = get_base_features(project_root_dir + "/sat_selector/training-set/RWS/Example_1.txt-b36f92fb.cnf")
    # with open(project_root_dir + "/sat_selector/" + "text.txt" , "w") as file:
    #     for k in sat_inst.features_dict :
    #         file.write(f"{k},")
    #     # for sat_solver in sat_solvers_in_pysat :
    #     print("Hello")
    #     file.write("best_solver_idx,")
    #     file.write("\n")

    # Get dataset

    cnf_path = project_root_dir + "/sat_selector/training-set/"
    process_pool = multiprocessing.Pool(processes=len(dataset_to_build))
    for dir in dataset_to_build:
        dataset_path = cnf_path + dir
        files = os.listdir(dataset_path)
        if len(files):
            process_pool.apply_async(building_thread, (files[:], dataset_path,))

    process_pool.close()
    process_pool.join()
