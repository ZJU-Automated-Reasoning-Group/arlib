'''
conjuctive思想，并利用unsat_core作为反馈信息，基本上有三种情况
1. unsat_core仅包含precond：precond可能与大部分cnt_list冲突，适合使用disjunctive_check
2. unsat_core仅包含cnt_list：有可能是cnt_list内部冲突，将它们划分为不同子集再处理
3. unsat_core包含precond&cnt_list：先处理冲突部分(unary_check)，接着处理剩下部分
'''
import z3
from typing import List

def unary_check(solver: z3.Solver, cnt_list: List[z3.ExprRef], res_label: List[int], check_list: List[int]):
    for i in check_list:
        solver.push()  
        solver.add(cnt_list[i])
        res = solver.check()
        if res == z3.sat:
            res_label[i] = 1
        elif res == z3.unsat:
            res_label[i] = 0
        solver.pop()

def disjunctive_check(solver: z3.Solver, cnt_list: List[z3.ExprRef], res_label: List[int], check_list: List[int]):
    conditions = [cnt_list[i] for i in check_list]

    if len(conditions) == 0:
        return  # 如果没有需要检查的条件，直接返回
    if len(conditions) == 1:
        f = conditions[0]
    else:
        f = z3.Or(conditions)

    solver.push()
    solver.add(f)
    s_res = solver.check()

    if s_res == z3.unsat:
        for i in check_list:
            res_label[i] = 0
        solver.pop()
    elif s_res == z3.sat:
        m = solver.model()
        new_check_list = []
        for i in check_list:
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                res_label[i] = 1
            else:
                new_check_list.append(i)
        solver.pop()
        disjunctive_check(solver, cnt_list, res_label, new_check_list)
    
def unsat_check_process(solver: z3.Solver, cnt_list: List[z3.ExprRef], res_label: List[int], check_list: List[int]):
    solver.push()
    
    # 检查所有（包括 precond 和 cnt_list）
    for i in check_list:
        solver.assert_and_track(cnt_list[i], str(i))
    
    check_result = solver.check()
    
    if check_result == z3.sat:
        # 所有 cnt_list 都是 SAT
        for i in check_list:
            res_label[i] = 1
        solver.pop()
    elif check_result == z3.unsat:
        # 找到 unsat_core
        unsat_core_indices = {int(str(c)) for c in solver.unsat_core()}
        unsat_set_indices = list(unsat_core_indices)
        sat_set_indices = [i for i in check_list if i not in unsat_core_indices]
        solver.pop()
        # if 最小冲突子集只有 precond
        if (-1 in unsat_set_indices) and (len(unsat_set_indices) == 1):
            # 用 disjunctive_check
            disjunctive_check(solver, cnt_list, res_label, check_list)
        # elif 最小冲突子集只有 cnt_list
        elif -1 not in unsat_set_indices:
            # 根据 unsat_core 划分集合，然后递归处理
            if len(unsat_set_indices) == 1:
                unary_check(solver, cnt_list, res_label, unsat_set_indices)
                unsat_check_process(solver, cnt_list, res_label, sat_set_indices)
            else:
                subsets = [[unsat_set_indices[i]] for i in range(len(unsat_set_indices))]
                # 将 sat_set_indices 平分到 subsets 中
                for i, sat_item in enumerate(sat_set_indices):
                    subsets[i % len(unsat_set_indices)].append(sat_item)
                for subset in subsets:
                    unsat_check_process(solver, cnt_list, res_label, subset)
        # else 最小冲突子集有 precond 和 cnt_list
        else:
            # 用 unary_check 处理 unsat_core，然后递归处理剩下的
            unary_check_list = unsat_set_indices
            unary_check_list.remove(-1)
            normal_check_list = [i for i in check_list if i not in unary_check_list]
            unary_check(solver, cnt_list, res_label, unary_check_list)
            unsat_check_process(solver, cnt_list, res_label, normal_check_list)

def unsat_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    res = [2] * len(cnt_list)
    solver = z3.Solver()
    solver.assert_and_track(precond, "-1")
    check_list = list(range(len(cnt_list))) 
    unsat_check_process(solver, cnt_list, res, check_list)
    return res

# # Test 1 [1, 1, 0]
# x = z3.Int('x')
# phi = z3.And(x >= 2, x <= 2)
# predicates = [x >= 2, x <= 2, x > 3] 
# result = unsat_check(phi, predicates)
# print("Result: ", result)
