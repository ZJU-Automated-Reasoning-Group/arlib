"""
Interface for CDCL(T)-based Solver
"""
# import itertools
from abc import ABC, abstractmethod
from arlib.cdclt.parallel_cdclt_process import parallel_cdclt_process
from arlib.cdclt.parallel_cdclt_thread import parallel_cdclt_thread
from arlib.cdclt.simple_cdclt import boolean_abstraction
from arlib.utils import SolverResult


class CDCLTSolver(ABC):
    """
    Abstract base class for a solver which implements the Conflict Driven Clause Learning (CDCL) algorithm for solving
    Satisfiability Modulo Theories (SMT) problems.
    """

    def __init__(self, **kwargs):
        self.seed = kwargs.get("seed", 1)

    @abstractmethod
    def solve_smt2_string(self, smt2string: str, logic: str) -> SolverResult:
        """
        Abstract method that solves an SMT-LIB2 problem with the CDCL algorithm.
        Parameters:
        -----------
        smt2string : str
            The SMT-LIB2 problem in its input format.
        logic : str
            The logic in use.
        Returns:
        --------
        SolverResult
            The result of the solver as a SolverResult object.
        """
        pass

    @abstractmethod
    def solve_smt2_file(self, filename: str, logic: str) -> SolverResult:
        pass


class SequentialCDCLTSolver(CDCLTSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.smt2_file = None

    def solve_smt2_string(self, smt2string: str, logic: str) -> SolverResult:
        pass

    def solve_smt2_file(self, filename: str, logic: str) -> SolverResult:

        pass


def dump_bool_skeleton(numeric_clauses, output_file: str):
    """
    Dump numerical clauses to a CNF file in output_file
    """
    if len(numeric_clauses) == 0:
        return

    num_var = -1
    for cls in numeric_clauses:
        num_var = max(num_var, max([abs(lit) for lit in cls]))

    header = ["p cnf {0} {1}".format(num_var, len(numeric_clauses))]
    with open(output_file, 'w+') as file:
        for info in header:
            file.write(info + "\n")
        for cls in numeric_clauses:
            file.write(" ".join([str(l) for l in cls]) + " 0\n")

class ParallelCDCLTSolver(CDCLTSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Mode
        -  process: process-based
        -  thread: thread-based
        -  preprocess: only perform preprocessing and dump the Boolean skeleton!
        """
        self.parallel_mode = kwargs.get("mode", "process")
        self.smt2_file = "tmp.smt2"

    def solve_smt2_string(self, smt2string: str, logic: str) -> SolverResult:
        if self.parallel_mode == "process":
            return parallel_cdclt_process(smt2string, logic)
        elif self.parallel_mode == "thread":
            return parallel_cdclt_thread(smt2string, logic)
        elif self.parallel_mode == "preprocess":
            cls = boolean_abstraction(smt2string)
            dump_bool_skeleton(cls, self.smt2_file+".cnf")
        else:
            return parallel_cdclt_process(smt2string, logic)

    def solve_smt2_file(self, filename: str, logic: str) -> SolverResult:
        self.smt2_file = filename
        smt2_file = open(filename, "r")
        smt2string = smt2_file.read()
        return self.solve_smt2_string(smt2string, logic)
