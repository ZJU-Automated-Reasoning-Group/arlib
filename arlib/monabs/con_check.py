"""
Conjunctive under-approximation
"""
import z3
from typing import List


def get_no_confict_sub_cnt_list_idx(cnt_list: List[z3.ExprRef]) -> List[List[z3.ExprRef]]:
    sub_cnt_list_idx = []
    solver = z3.Solver()

    def process_subsets(current_subset: List[z3.ExprRef]) -> bool:
        solver.push()
        for i, idx in enumerate(current_subset):
            solver.assert_and_track(cnt_list[idx], str(idx))
        solver_result = solver.check()
        solver.pop()
        if solver_result == z3.sat:
            sub_cnt_list_idx.append(current_subset)
        elif solver_result == z3.unsat:
            # get unsat core
            unsat_core_indices = {int(str(c)) for c in solver.unsat_core()}
            unsat_set_indices = list(unsat_core_indices)
            sat_set_indices = [i for i in current_subset if i not in unsat_core_indices]
            subsets = [[unsat_set_indices[i]] for i in range(len(unsat_set_indices))]
            # Distribute sat_set_indices among unsat_set_indices
            for i, sat_item in enumerate(sat_set_indices):
                subsets[i % len(unsat_set_indices)].append(sat_item)
            for subset in subsets:
                process_subsets(subset)
            
    process_subsets(range(len(cnt_list)))
    return sub_cnt_list_idx


def unary_check_cached(precond: z3.ExprRef, cnt_list: List[z3.ExprRef], results: List[int], check_list: List[int]):
    for i in check_list:   
        if results[i] is not None:
            continue
        solver = z3.Solver()
        solver.add(precond)
        solver.add(cnt_list[i])  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            results[i] = 1
            for j in check_list:
                if results[j] is None and z3.is_true(model.eval(cnt_list[j], model_completion=True)):
                    results[j] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2


def unary_check_incremental_cached(solver: z3.Solver, cnt_list: List[z3.ExprRef], results: List[int], check_list: List[int]):
    for i in check_list:   
        if results[i] is not None:
            continue
        solver.push()  # Save the current state        
        solver.add(cnt_list[i])  # Add the current constraint
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            results[i] = 1
            for j in check_list:
                if results[j] is None and z3.is_true(model.eval(cnt_list[j], model_completion=True)):
                    results[j] = 1
        elif res == z3.unsat:
            results[i] = 0
        else:
            results[i] = 2
        solver.pop()  # Restore the state
        

def disjunctive_check_incremental_cached(solver: z3.Solver, cnt_list: List[z3.ExprRef], results: List[int], check_list: List[int]):
    f = z3.BoolVal(False)
    conditions = [cnt_list[i] for i in check_list if results[i] is None]
    
    if len(conditions) == 0:
        return

    f = z3.Or(conditions)

    if z3.is_false(f):
        return
    
    solver.push()
    solver.add(f)
    res = solver.check()
    if res == z3.unsat:
        for i in check_list:
            results[i] = 0
        solver.pop()
    elif res == z3.sat:
        m = solver.model()
        solver.pop()
        new_check_list = []
        for i in check_list:
            if results[i] is None and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                results[i] = 1
            elif results[i] is None:
                new_check_list.append(i)
        disjunctive_check_incremental_cached(solver, cnt_list, results, new_check_list)

 
def conjunctive_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef], alogorithm: int = 0) -> List:
    """
    Perform a conjunctive satisfiability check on a list of constraints under a given precondition.
    This function checks whether the conjunction of a set of constraints (`cnt_list`) is satisfiable 
    under a given precondition (`precond`). It uses a Z3 solver to perform the satisfiability checks 
    and supports different algorithms for handling unsatisfiable cores.
    Args:
        precond (z3.ExprRef): The precondition to be added to the solver.
        cnt_list (List[z3.ExprRef]): A list of Z3 expressions representing the constraints to be checked.
        alogorithm (int): The algorithm to use for handling unsatisfiable cores. 
                          Options are:
                          - 0: Unary check with caching.
                          - 1: Incremental unary check with caching.
                          - 2: Incremental disjunctive check with caching.
    Returns:
        List: A list of results where each index corresponds to the satisfiability of the respective 
              constraint in `cnt_list`. A value of `1` indicates satisfiable, `0` indicates unsatisfiable,
              and `2` indicates unknown.
    Notes:
        - The function uses `get_no_confict_sub_cnt_list_idx` to partition the constraints into 
          non-conflicting subsets.
        - Unsatisfiable cores are handled by moving them to a waiting list for further processing 
          based on the selected algorithm.
    """
    
    results = [None] * len(cnt_list)
    solver = z3.Solver()
    sub_cnt_list_idx = get_no_confict_sub_cnt_list_idx(cnt_list)
    waiting_list_idx = []

    solver.add(precond)

    for i, idx_list in enumerate(sub_cnt_list_idx):
        solver.push()
        # Check if conjunction of all predicates is satisfiable
        for idx in idx_list:
            solver.assert_and_track(cnt_list[idx], str(idx))

        while True:
            solver_result = solver.check()
            solver.pop()
            
            if solver_result == z3.sat:
                # All constraints are satisfiable
                for idx in idx_list:
                    results[idx] = 1
                break
            else:
                # Move unsat core to waiting list
                unsat_core = solver.unsat_core()
                for idx in unsat_core:
                    idx_list.remove(int(str(idx)))
                    waiting_list_idx.append(int(str(idx)))

                if len(idx_list) == 0:
                    break
                
                solver.push()
                for idx in idx_list:
                    solver.assert_and_track(cnt_list[idx], str(idx))
        
        if alogorithm == 0:
            unary_check_cached(precond, cnt_list, results, waiting_list_idx)
        elif alogorithm == 1:
            unary_check_incremental_cached(solver, cnt_list, results, waiting_list_idx)
        elif alogorithm == 2:
            disjunctive_check_incremental_cached(solver, cnt_list, results, waiting_list_idx)
        else:
            raise ValueError("Invalid algorithm choice. Choose 0, 1, or 2.")
 
    return results
