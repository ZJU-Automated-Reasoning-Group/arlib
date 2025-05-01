"""
A solver for SC-reduction, specifically using the DTC-based approach
"""

from math import pi, sqrt, floor, ceil, log
from collections import defaultdict
from copy import deepcopy


class SCSolver:
    def __init__(self, mode):
        self.mode = mode

    # function to estimate the number of quantum iterations
    def estimate(self, sol, cand):
        res = floor(sqrt(cand))
        if sol == 0:
            return res
        it = sol
        while it > 0:
            # according to https://doi.org/10.1137/050644719,
            # one target is returned after at most 0.9\sqrt{N/M} queries,
            # where N is the search space size and M is the number of targets.
            res += floor(0.9*sqrt(cand/it))
            it -= 1
        return res

    def solve(self, graph, grammar):
        if self.mode == "Cubic":
            self.__cubic_solve(graph, grammar)
    
    # convert the CFL-reachability to SC-reduction
    # according to https://doi.org/10.1109/LICS.1997.614960
    def convert(self, graph, grammar):
        constraint = defaultdict(lambda: defaultdict(set))
        set_variable = set()
        # process graph
        for v in graph.ds_structure.vertices:
            index = str(graph.ds_structure.edge_indices[v])
            constraint['con0']['X'+index].add('X'+index+','+'node'+index)
            set_variable.add('X'+index)
        for p, elist in graph.ds_structure.symbol_pair.items():
            for e in elist:
                index1 = str(graph.ds_structure.edge_indices[e[0]])
                index2 = str(graph.ds_structure.edge_indices[e[1]])
                constraint['con1']['X'+index1].add('X'+index1+','+p+','+'X'+index2)
        for left,v in grammar.items():
            for right in v:
                if len(right) == 2:
                    for i in graph.ds_structure.edge_indices.values():
                        constraint['pro']['X'+str(i)].add('Rchd'+right[0]+str(1)+str(i)+','+right[0]+str(1)+','+'X'+str(i))
                        constraint['pro']['Rchd'+right[0]+str(1)+str(i)].add('Dst'+left+str(i)+','+right[1]+str(1)+','+'Rchd'+right[0]+str(1)+str(i))
                        constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'Dst'+left+str(i))
                        set_variable.add('Rchd'+right[0]+str(1)+str(i))
                        set_variable.add('Dst'+left+str(i))
                elif len(right) == 1:
                    if right[0] == 'ε':
                        for i in graph.ds_structure.edge_indices.values():
                            constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'X'+str(i))
                    else:
                        for i in graph.ds_structure.edge_indices.values():
                            constraint['con1']['X'+str(i)].add('X'+str(i)+','+left+','+'Dst'+left+str(i))
                            constraint['pro']['X'+str(i)].add('Dst'+left+str(i)+','+right[0]+str(1)+','+'X'+str(i))
                            set_variable.add('Dst'+left+str(i))
                else:
                    raise('grammar error')
        # save the constraint info
        self.constraint = constraint
        self.set_variable = set_variable
        self.nnn = len(constraint['con1'])

    # DTC-based SC-reduction
    def __cubic_solve(self, graph, grammar):
        # convert the graph to constraints
        # using https://doi.org/10.1109/LICS.1997.614960
        self.convert(graph,grammar)
        constraint = deepcopy(self.constraint)

        # step 1
        W = set()
        for k in constraint['con0']:
            for i in constraint['con0'][k]:
                left, right = i.split(',')
                W.add(('con', left, right))
        # step 2
        ground = dict()
        for i in self.set_variable:
            ground[i] = 0
        # number of classical iteraitons
        whole_iteration = 0
        # number of quantum iterations
        gs_iteration = 0

        # start the worklist analysis of SC-reduction
        while len(W) > 0:
            a = W.pop()
            if a[0] == 'con':
                if len(a) < 4:
                    X = a[1]
                    iteration = 0
                    solution = 0
                    for i in constraint['norm'][X]:
                        iteration+=1
                        whole_iteration+=1
                        l, r = i.split(',')
                        s = l+','+a[2]
                        if s not in constraint['con0'][l]:
                            solution+=1
                            constraint['con0'][l].add(s)
                            W.add(('con', l, a[2]))
                    gs_iteration+=self.estimate(solution,iteration)
                else:
                    X, V = a[1], a[3]
                    iteration = 0
                    solution = 0
                    for i in constraint['pro'][X]:
                        iteration+=1
                        whole_iteration+=1
                        l,m,r = i.split(',')
                        s = l+','+a[3]
                        if s not in constraint['norm'][V]:
                            solution+=1
                            constraint['norm'][V].add(s)
                            W.add(('norm', l, a[3]))
                    gs_iteration+=self.estimate(solution,iteration)
                    iteration = 0
                    solution = 0
                    for i in constraint['norm'][X]:
                        whole_iteration+=1
                        iteration += 1
                        l, r = i.split(',')
                        s = l+','+a[2]+','+a[3]
                        if s not in constraint['con1'][l]:
                            solution += 1
                            constraint['con1'][l].add(s)
                            W.add(('con', l, a[2], a[3]))
                    gs_iteration+=self.estimate(solution, iteration)

                        
            elif a[0] == 'norm':
                X, Y = a[1], a[2]
                tmp_set = set()
                iteration = 0
                solution = 0
                for i in constraint['con1'][Y]:
                    l,m,r = i.split(',')
                    if ground[r] == 1:
                        whole_iteration+=1
                        iteration += 1
                        s = a[1]+','+m+','+r
                        if s not in constraint['con1'][X]:
                            solution += 1
                            tmp_set.add(s)
                            W.add(('con', a[1], m, r))
                # update number of iterations
                gs_iteration += self.estimate(solution, iteration)
                for i in tmp_set:
                    constraint['con1'][X].add(i)
            else:
                raise('wrong type error')
            X = a[1]
            if ground[a[1]] == 0:
                ground[a[1]] = 1
                for k in self.constraint['con1']:
                    for i in constraint['con1'][k]:
                        l,m,r = i.split(',')
                        if ground[r] == 1:
                            if r == X:
                                W.add(('con', l, m, r))
                for i in self.constraint['norm'][X]:
                    l,r = i.split(',')
                    W.add(('norm', l, r))
        # print estimation result
        print('the number of classical iterations: ', whole_iteration)
        print('the number of quantum iterations: ', gs_iteration * 3)
        

        