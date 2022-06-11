# coding: utf-8

"""
Parallel DPLL(T) by separating PropSolver and Theory Solver?

- How to share the "learned clauses" or the blocking formulas, especially when
  one worker finishes while other workers are still working
- Maybe maintain a shared lemma database; when a worker starts to do sth new, it can "fetch" the clauses..

"""

import time
import multiprocessing
import z3
import re
from z3.z3util import get_vars
from .smtlib_solver import SMTLIBSolver
from .util import convert_value, RE_GET_EXPR_VALUE_ALL
from .config import m_smt_solver_bin


def mk_lit(m, p):
    if z3.is_true(m.eval(p, True)):
        return p
    else:
        return z3.Not(p)


index = 1


def abstract_atom(atom2bool, atom):
    global index
    if atom in atom2bool:
        return atom2bool[atom]
    p = z3.Bool("p%d" % index)
    index += 1
    atom2bool[atom] = p
    return p


def abstract_lit(atom2bool, lit):
    if z3.is_not(lit):
        return z3.Not(abstract_atom(atom2bool, lit.arg(0)))
    return abstract_atom(atom2bool, lit)


def abstract_clause(atom2bool, clause):
    return z3.Or([abstract_lit(atom2bool, lit) for lit in clause])


def abstract_clauses(atom2bool, clauses):
    return [abstract_clause(atom2bool, clause) for clause in clauses]


def solve_with_smtlib_solver(solver, lits):
    smt2sting = "(set-logic QF_BV)\n"
    smt2sting += "\n".join(solver.to_smt2().split("\n")[:-2])  # for removing (check-sat)
    check_cmd = "\n(check-sat-assuming ("
    for lit in lits:
        if z3.is_not(lit):
            check_cmd += "(not {})".format(str(lit.arg(0)))
        else:
            check_cmd += str(lit)
        check_cmd += " "
    check_cmd += "))"
    smt2sting += check_cmd

    bin_solver = SMTLIBSolver(m_smt_solver_bin)
    res = bin_solver.check_sat_from_scratch(smt2sting)
    ret = "unknown"
    if res == "sat":
        print(bin_solver.get_expr_values(["p1", "p0", "p2"]))
        ret = z3.sat
    elif res == "unsat":
        ret = z3.unsat
    bin_solver.stop()
    return ret


def solver_worker_api(thcnt, propcnt, bool_vars, blocking_clause_queue, prop_model_queue, result_queue):
    """CDCL(T) worker
    :param thcnt: the theory constraints (p1 = atom1 and p2 = atom2 and ....)
    :param propcnt: the initial Boolean constraints (encoding the Boolean structure)
    :param bool_vars
    :param blocking_clause_queue: for refining Boolean constraints
    :param prop_model_queue: to be checked by theory solver
    """
    # TODO: use pysat
    prop_solver = z3.Solver()
    bool_fml = z3.And(z3.parse_smt2_string(propcnt))
    z3bool_vars = get_vars(bool_fml)
    prop_solver.add(bool_fml)
    # print(prop_solver.dimacs()) # is this API "reliable"?

    # pysat_solver = SATSolver()
    # pysat_solver.add_clauses_from_string(prop_solver.dimacs())

    theory_solver = z3.SolverFor("QF_BV")
    theory_solver.add(z3.And(z3.parse_smt2_string(thcnt)))
    final_res = "unknown"

    while True:
        is_sat = prop_solver.check()
        if z3.sat == is_sat:
            m = prop_solver.model()
            lits = []
            for var in z3bool_vars:
                if m.eval(var, True):
                    lits.append(var)
                else:
                    lits.append(z3.Not(var))
            if z3.unsat == theory_solver.check(lits):
                # FIXME: use the naive "blocking formula" or use unsat core to refine
                # If unsat_core is enabled, the bit-vector solver might be slow
                # prop_solver.add(Not(And(theory_solver.unsat_core())))
                prop_solver.add(z3.Not(z3.And(lits)))
            else:
                # print(theory_solver.model())
                final_res = "sat"
                break
        elif z3.unsat == is_sat:
            final_res = "unsat"
            break
        else:
            break
    result_queue.put(final_res)


def solver_worker(thcnt, propcnt, bool_vars, blocking_clause_queue, prop_model_queue, result_queue):
    """CDCL(T) worker
    :param thcnt: the theory constraints (p1 = atom1 and p2 = atom2 and ....)
    :param propcnt: the initial Boolean constraints (encoding the Boolean structure)
    :param bool_vars
    :param blocking_clause_queue: for refining Boolean constraints
    :param prop_model_queue: to be checked by theory solver
    """
    theory_solver = SMTLIBSolver(m_smt_solver_bin)
    prop_solver = SMTLIBSolver(m_smt_solver_bin)

    theory_solver.assert_assertions(propcnt)
    prop_solver.assert_assertions(thcnt)

    # TODO: do the CEGAR loop (finish first version)
    final_res = "unknown"
    while True:
        ppsolver_res = prop_solver.check_sat()
        if ppsolver_res == "unsat":
            final_res = "unsat"
            break
        elif ppsolver_res == "sat":
            # get the model, translate to py values (is this necessary?)
            model_str = prop_solver.get_expr_values(bool_vars)
            model_strpy = re.findall(RE_GET_EXPR_VALUE_ALL, model_str)
            model = {value[0]: convert_value(value[1]) for value in model_strpy}
            assumption_lits = []  # to be checked by the theory solver
            for atom in model:
                if model[atom]:
                    assumption_lits.append(atom)
                else:
                    assumption_lits.append("(not {})".format(atom))
            # print("assumptions: ", assumption_literals)
            thsolver_res = theory_solver.check_sat_assuming(assumption_lits)
            if "unsat" == thsolver_res:
                blocking_clauses = "(assert (not (and " + " ".join(assumption_lits) + ")))\n"
                # print("blocking clause: ", all_expressions_str)
                prop_solver.assert_assertions(blocking_clauses)
            elif "sat" == thsolver_res:
                final_res = "sat"
                break
            else:
                break  # is this OK?
        else:
            break  # is this OK?
    theory_solver.stop()
    prop_solver.stop()
    result_queue.put(final_res)


g_process_queue = []



def simple_cdclT(clauses):
    prop_solver = z3.SolverFor("QF_FD")
    theory_solver = z3.SolverFor("QF_BV")
    abs = {}
    prop_solver.add(abstract_clauses(abs, clauses))
    theory_solver.add([p == abs[p] for p in abs])
    while True:
        is_sat = prop_solver.check()
        if z3.sat == is_sat:
            m = prop_solver.model()
            lits = [mk_lit(m, abs[p]) for p in abs]
            if z3.unsat == theory_solver.check(lits):
                # FIXME: use the naive "blocking formula" or use unsat core to refine
                # If unsat_core is enabled, the bit-vector solver might be slow
                # prop_solver.add(Not(And(theory_solver.unsat_core())))
                prop_solver.add(z3.Not(z3.And(lits)))
                # print(prop_solver)
            else:
                # print(theory_solver.model())
                print("sat")
                return
        else:
            # print(is_sat)
            print("unsat")
            return

def main_par(fml):
    global g_process_queue
    # set_param("verbose", 15)
    # first, to CNF
    # TODO: This is ugly
    # E.g., solve-eqs, qe-lite, ...
    tac = z3.Then(z3.Tactic("simplify"), z3.Tactic("tseitin-cnf"))
    clauses = []
    for cls in tac(fml)[0]:
        if z3.is_or(cls):
            tmp_cls = []
            for lit in cls.children():
                tmp_cls.append(lit)
            clauses.append(tmp_cls)
        else:
            # unary clause
            clauses.append([cls])
    print("to cnf success!")

    prop_solver = z3.Solver()
    theory_solver = z3.Solver()
    atom2bool = {}

    # collect the constraints
    prop_solver.add(abstract_clauses(atom2bool, clauses))
    theory_solver.add([p == atom2bool[p] for p in atom2bool])

    thsolver_smt2sting = "(set-logic QF_BV)\n"
    thsolver_smt2sting += "\n".join(theory_solver.to_smt2().split("\n")[:-2])  # for removing (check-sat)

    ppsolver_smt2sting = "(set-logic QF_UF)\n"
    ppsolver_smt2sting += "\n".join(prop_solver.to_smt2().split("\n")[:-2])  # for removing (check-sat)

    # print(prop_solver.dimacs())
    # t = prop_solver.dimacs()
    # print(prop_solver)

    bool_vars = []
    for atom in atom2bool:
        bool_vars.append(str(atom2bool[atom]))

    start = time.time()
    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()
        blocking_clause_queue = multiprocessing.Queue()
        prop_model_queue = multiprocessing.Queue()
        print("workers: ", multiprocessing.cpu_count())

        for _ in range(multiprocessing.cpu_count()):
            g_process_queue.append(multiprocessing.Process(target=solver_worker_api,
                                                           args=(thsolver_smt2sting,
                                                                 ppsolver_smt2sting,
                                                                 bool_vars,
                                                                 blocking_clause_queue,
                                                                 prop_model_queue,
                                                                 result_queue
                                                                 )))
        # Start all
        for p in g_process_queue:
            p.start()

        try:
            # Wait at most 300 seconds for a return
            result = result_queue.get(timeout=600)
        except multiprocessing.queues.Empty:
            result = "unknown"
        for p in g_process_queue:
            p.terminate()

        print(result)

    print(time.time() - start)
