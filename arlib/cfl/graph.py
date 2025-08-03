
"""
A graph representation, specifically for the CFL-reachability problem
"""

from typing import List, Dict, Any, Union, Tuple, Optional, Set
from arlib.cfl.matrix import Matrix
from arlib.cfl.pag_matrix import PAG_Matrix


class Graph:
    def __init__(self, source_file: str, ds_mode: str) -> None:
        self.source_file: str = source_file
        self.ds_mode: str = ds_mode
        if ds_mode == "Matrix":
            self.ds_structure = Matrix(source_file)
        elif ds_mode == "PAG_Matrix":
            self.ds_structure = PAG_Matrix(source_file)
        else:
            raise Exception("This is not a valide ds_mode, ds_mode including Matrix")

    def add_vertex(self, vertex: Any) -> bool:
        return self.ds_structure.add_vertex(vertex)

    def add_edge(self, u: Any, v: Any, label: str) -> bool:
        return self.ds_structure.add_edge(u, v, label)

    def output_edge(self) -> List[List[Union[str, Any]]]:
        return self.ds_structure.output_edge()

    def check_edge(self, u: Any, v: Any, lable: str) -> bool:
        return self.ds_structure.check_edge(u, v, lable)

    def new_check_edge(self, u: Any, v: Any, lable: str) -> bool:
        return self.ds_structure.new_check_edge(u, v, lable)

    def output_set(self) -> List[Tuple[Any, Any]]:
        return self.ds_structure.output_set()

    def get_vertice(self) -> List[str]:
        return list(self.ds_structure.vertices.keys())

    def symbol_pair_l(self, label: str) -> List[Tuple[Any, Any]]:
        return self.ds_structure.symbol_pair[label]

    def symbol_pair(self) -> Dict[str, List[Tuple[Any, Any]]]:
        return self.ds_structure.symbol_pair

    def dump_dot(self) -> None:
        return self.ds_structure.dump_dot()

    def dump_dot1(self) -> None:
        return self.ds_structure.dump_dot1()
