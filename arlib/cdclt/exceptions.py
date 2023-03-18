from arlib.utils.exceptions import *


class TheorySolverSuccess(SMTSuccess):
    """
    The theory solver checks T-consistency successfully
    """


class TheorySolverError(SMTError):
    """
    The theory solver encounters an error while checking T-consistency
    """


class BoolSolverSuccess(SMTSuccess):
    """
    The boolean solver checks satisfiability successfully
    In the CDDL(T) settings, it means that the Boolean skeleton is unsatisfiable (
    Thus, the orignal SMT formula is also unsatisfiable
    )
    """



class BoolSolverError(SMTError):
    """
    """


class SimplifierSuccess(SMTSuccess):
    """
    TBD
    """


class SimplifierError(SMTError):
    """
    """


class PySMTSolverError(SMTError):
    """
    TBD
    """


