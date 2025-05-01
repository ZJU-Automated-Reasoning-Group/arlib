from math import pi, sqrt, floor, ceil, log


class CFLSolver:
    def __init__(self, mode):
        self.mode = mode

    # function to estimate the number of quantum iterations
    def estimate(self, sol, cand, nnn):
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
        return floor(res * log(nnn))

    def solve(self, graph, grammar):
        if self.mode == "Cubic":
            self.__cubic_solve(graph, grammar)
    
    # DTC-based CFL-reachability
    def __cubic_solve(self, graph, grammar):
        # each edge in worklist stand by [(edge, node, node)]
        # number of classical iterations
        whole_iteration=0
        # number of quantum iterations
        gs_iteration=0
        print('graph size: ',len(graph.ds_structure.vertices))
        nnn = len(graph.ds_structure.vertices)
        Worklist = graph.output_edge()
        for nullable_variable in grammar.epsilon:
            for node in graph.get_vertice():
                graph.add_edge(node, node, nullable_variable)
                Worklist.append([nullable_variable,node,node])
        # worklist analysis of DTC-based CFL-reachability
        while Worklist != []:
            selected_edge = Worklist.pop()
            for X, right_list in grammar.items():
                # X: key: variable right_list : list of all right handside of production
                for right in right_list:
                    # X = Y
                    if len(right) == 1 and right[0] == selected_edge[0]:
                        Y = right[0]
                        for pair in graph.symbol_pair_l(Y):
                            # O(n) for graph.symbol_pair_l return list of node pair
                            if not graph.new_check_edge(pair[0],pair[1],X):
                                # O(m) m stand for len(varibale, terminal) 
                                graph.add_edge(pair[0],pair[1],X)
                                Worklist.append([X,pair[0],pair[1]])
            # codes that lead to cubic bottleneck
            for X, right in grammar.items():
                for right_symbols in right:
                    if len(right_symbols) == 2 and right_symbols[0] == selected_edge[0]:
                        Y = right_symbols[0]
                        Z = right_symbols[1]
                        if Z in graph.symbol_pair():
                            num_of_sol=0
                            iteration=0
                            for pair in graph.symbol_pair_l(Z):
                                iteration+=1
                                j = selected_edge[2]
                                i = selected_edge[1]
                                k = pair[1]
                                if pair[0] == selected_edge[2]:
                                    if not (graph.new_check_edge(i,k,X)):
                                        graph.add_edge(i,k,X)
                                        Worklist.append([X,i,k])
                                        num_of_sol+=1
                            # update number of classical and quantum iterations
                            gs_iteration+=self.estimate(num_of_sol,iteration,nnn)
                            whole_iteration+=iteration
            for X, right in grammar.items():
                for right_symbols in right:
                    if len(right_symbols) == 2 and right_symbols[1] == selected_edge[0]:
                        Y = right_symbols[1]
                        Z = right_symbols[0]
                        if Z in graph.symbol_pair():
                            num_of_sol=0
                            iteration=0
                            for pair in graph.symbol_pair_l(Z):
                                iteration+=1
                                j = selected_edge[2]
                                i = selected_edge[1]
                                k = pair[0]
                                if pair[1] == i:
                                    if not (graph.new_check_edge(k,j,X)):
                                        graph.add_edge(k,j,X)
                                        Worklist.append([X,k,j])
                                        num_of_sol+=1
                            # update number of classical and quantum iterations
                            gs_iteration+=self.estimate(num_of_sol,iteration,nnn)
                            whole_iteration+=iteration
        # print estimation result
        print('the number of classical iterations: ', whole_iteration)
        print('the number of quantum iterations: ', gs_iteration * 3)
        