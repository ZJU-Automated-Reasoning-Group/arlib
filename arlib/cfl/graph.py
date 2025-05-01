
"""
A graph representation, specifically for the CFL-reachability problem
"""

from arlib.cfl.matrix import Matrix
from arlib.cfl.pag_matrix import PAG_Matrix


class Graph:
    def __init__(self, source_file, ds_mode):
        self.source_file = source_file
        self.ds_mode = ds_mode
        if ds_mode == "Matrix":
            self.ds_structure = Matrix(source_file)
        elif ds_mode == "PAG_Matrix":
            self.ds_structure = PAG_Matrix(source_file)
        else:
            raise Exception("This is not a valide ds_mode, ds_mode including Matrix")
    
    def add_vertex(self, vertex):
        return self.ds_structure.add_vertex(vertex)

    def add_edge(self, u, v, label):
        return self.ds_structure.add_edge(u, v, label)
    
    def output_edge(self):
        return self.ds_structure.output_edge()
    
    def check_edge(self, u, v, lable):
        return self.ds_structure.check_edge(u, v, lable)
    
    def new_check_edge(self, u, v, lable):
        return self.ds_structure.new_check_edge(u, v, lable)

    def output_set(self):
        return self.ds_structure.output_set()
    
    def get_vertice(self):
        return self.ds_structure.vertices.keys()
    
    def symbol_pair_l(self, label):
        return self.ds_structure.symbol_pair[label]
    
    def symbol_pair(self):
        return self.ds_structure.symbol_pair
    
    def dump_dot(self):
        return self.ds_structure.dump_dot()

    def dump_dot1(self):
        return self.ds_structure.dump_dot1()
    