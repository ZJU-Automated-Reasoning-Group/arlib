"""
Conjunctive under-approximation
TODO: check for correctness
      use some maximal satisfying subsets algo.
"""
from typing import List
import z3


def intersection_based_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
    """
    Check satisfiability of each predicate in cnt_list using intersection-based method.

    Conjunction Check:
    - Adds the precondition precond to the solver.
    - Constructs all_conj, the conjunction of all constraints in cnt_list.
    - Checks if all_conj is satisfiable:
      - If satisfiable, sets all results to 1 (satisfiable).
      - Otherwise, proceeds to find minimal satisfiable subsets.

    Refinement Loop:
    - Initializes S as a set of indices of cnt_list.
    - Iteratively checks subsets of S:
        - For each index idx in S, constructs a subset excluding idx.
        - Checks satisfiability of this subset plus the precondition.
        - If satisfiable, marks the constraint at idx as satisfiable and removes idx from S.
    - Continues until no more indices can be removed.
    """
    results = [0] * len(cnt_list)
    solver = z3.Solver()

    # Add precondition
    solver.add(precond)

    # Check if conjunction of all predicates is satisfiable
    all_conj = z3.And(cnt_list)
    solver.push()
    solver.add(all_conj)
    if solver.check() == z3.sat:
        # All constraints are satisfiable
        results = [1] * len(cnt_list)
    else:
        # Find minimal satisfiable subsets
        S = set(range(len(cnt_list)))
        while S:
            found = False
            for idx in S.copy():
                solver.push()
                subset_conj = z3.And([cnt_list[i] for i in S if i != idx])
                solver.add(subset_conj)
                if solver.check() == z3.sat:
                    results[idx] = 1
                    S.remove(idx)
                    found = True
                solver.pop()
            if not found:
                break

    solver.pop()
    return results
