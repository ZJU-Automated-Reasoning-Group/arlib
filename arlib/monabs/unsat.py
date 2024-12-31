"""
先处理unsat_core给出的最小冲突子集，剩下的很有可能是sat，如果已经没有可利用的最小冲突子集，
但是剩下的不是sat，那么就用一般方法检测剩下的
"""
import z3
from typing import List
from itertools import chain, combinations

def unsat_process(solver: z3.Solver, cnt_list: List[z3.ExprRef], res_label: List[int]):
    flag = False
    
    # 先把还没处理的约束加进去
    solver.push()
    for i, label in enumerate(res_label):
        if label == 2:
            solver.assert_and_track(cnt_list[i], str(i))
            flag = True
    
    # 如果没有待处理的约束，直接返回
    if not flag:
        solver.pop()
        return
    
    # 检查是否不满足
    s_res = solver.check()

    # 不满足
    if s_res == z3.unsat:
        # 获取最小冲突子集
        unsat_core = solver.unsat_core()
        unsat_core_str = [str(c) for c in unsat_core]
        solver.pop()
        
        # 没有可利用的最小冲突子集，但是还有没有检测完的，用一般方法来检测
        if not unsat_core_str:
            for i in range(len(res_label)):
                if res_label[i] == 2:
                    solver.push()  
                    solver.add(cnt_list[i])  
                    res = solver.check()
                    if res == z3.sat:
                        res_label[i] = 1
                        # m = solver.model()
                        # for i in range(len(res_label)):
                        #     if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                        #         res_label[i] = 1
                    elif res == z3.unsat:
                        res_label[i] = 0
                    solver.pop()  
            return
        
        # 有可利用的最小冲突子集，先把这些进行一般处理
        for i in unsat_core_str:
            i = int(i)
            solver.push()
            solver.add(cnt_list[i])
            res = solver.check()
            if res == z3.sat:
                res_label[i] = 1
                # m = solver.model()
                # for i in range(len(res_label)):
                #     if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], model_completion=True)):
                #         res_label[i] = 1
            elif res == z3.unsat:
                res_label[i] = 0
            solver.pop()
    # 剩下的全部满足
    elif s_res == z3.sat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 1
        solver.pop()
        return
    
    unsat_process(solver, cnt_list, res_label)

def unsat_check(precond: z3.ExprRef, cnt_list: List[z3.ExprRef]) -> List[int]:
    res = [2] * len(cnt_list)
    solver = z3.Solver()
    solver.set(unsat_core=True)
    solver.add(precond)
    unsat_process(solver, cnt_list, res)
    return res

# # Test 1 [1, 1, 0]
# x = z3.Int('x')
# phi = z3.And(x >= 2, x <= 2)
# predicates = [x >= 2, x <= 2, x > 3, 2<-1] 
# # predicates = [x >= 2, x <= 2, x > 3] 
# result = unsat_check(phi, predicates)
# print(result)
