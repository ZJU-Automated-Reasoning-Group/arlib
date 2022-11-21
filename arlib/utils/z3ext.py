
from typing import List
import z3


def compact_check_misc(precond, cnt_list, res_label, models):
    """
    TODO: In our settings, as long as there is one unsat, we can stop
      However, this algorithm stops until "all the remaining ones are UNSAT (which can have one of more instances)"
    """
    f = z3.BoolVal(False)
    for i in range(len(res_label)):
        if res_label[i] == 2: f = z3.Or(f, cnt_list[i])
    if z3.is_false(f): return

    sol = z3.SolverFor("QF_BV")
    g = z3.And(precond, f)
    sol.add(g)
    s_res = sol.check()
    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2: res_label[i] = 0
    elif s_res == z3.sat:
        m = sol.model()
        models.append(m)  # counterexample
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], True)):
                res_label[i] = 1
    else:
        return
    compact_check_misc(precond, cnt_list, res_label, models)


