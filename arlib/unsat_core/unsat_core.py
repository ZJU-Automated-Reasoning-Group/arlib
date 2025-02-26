"""
Provide external intearface for computing unsat cores that encapsulates
marco.py, musx.py, optux.py (and other possible new implementations)

NOTICE: distinguish the following types of problems:
    (1) unsat core: a list of literals that is unsatisfiable
    (2) minimal unsatisfiable subset (MUS): a subset of literals that is unsatisfiable
    (3) MUS enumeration: enumerating all MUSs
    ...?
"""

