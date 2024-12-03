"""
Conjunctive under-approximation
可能的改进方法:
1. unsat_core
2. 启发式 (如 dp / greedy / binary search)
3. 其他
"""

from itertools import chain, combinations
from typing import List

import z3


# Generate all non-empty subsets of an iterable, ordered by subset size from largest to smallest
def all_subsets(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable), 0, -1))


def intersection_based_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List:
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
        tested_subsets = set()
        S = {i: cnt_list[i] for i in range(len(cnt_list))}
        solver.pop()
        # If not all constraints are satisfiable, try to find minimal satisfiable subsets
        # TODO: 这里是暴力解法，直接生成所有可能的子集进行遍历 (ordered by subset size from largest to smallest)
        while S:
            subsets = all_subsets(S)
            for subset in subsets:
                if subset in tested_subsets:
                    continue
                tested_subsets.add(subset)
                solver.push()
                subset_conj = z3.And([cnt_list[i] for i in subset])
                solver.add(subset_conj)
                if solver.check() == z3.sat:
                    for idx in subset:
                        results[idx] = 1
                        del S[idx]
                    solver.pop()
                    break
                solver.pop()
            else:
                # If no satisfiable subset was found in the remaining set, exit the loop
                break

    return results

# # Test 1 [1, 1, 0]
# x = z3.Int('x')
# phi = z3.And(x >= 2, x <= 2)
# predicates = [x >= 2, x <= 2, x > 3]
# result = intersection_based_check(phi, predicates)
# print(result)

# # Test 2 [1, 0, 0]
# x, y = z3.Reals('x y')
# phi = x>100
# f1 = x > 0; f2 = z3.And(x > y, x < y); f3 = x < 3
# predicates = [f1, f2, f3]
# result = intersection_based_check(phi, predicates)
# print(result)
