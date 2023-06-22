"""
Factory for OMT solvers
"""

# from pysmt.factory import Factory
# from pysmt.environment import Environment
from pysmt.logics import QF_UFLIRA, LRA, QF_UFLRA, QF_LIA
from pysmt.exceptions import (NoSolverAvailableError, SolverRedefinitionError,
                              NoLogicAvailableError,
                              SolverAPINotFound)

DEFAULT_OMT_SOLVERS = {'Optimizer': ['optimsat', 'z3',
                                     'msat_incr', 'optimsat_incr', 'yices_incr', 'z3_incr',
                                     'msat_sua', 'optimsat_sua', 'yices_sua', 'z3_sua'
                                     ]}

DEFAULT_OPTIMIZER_LOGIC = QF_LIA


class OMTFactory:

    def __int__(self):
        # self._env = env
        print("XXXXXXXX")
        self._all_optimizers = {}
        self._default_optimizer_logic = DEFAULT_OPTIMIZER_LOGIC
        # self.preferences = dict(DEFAULT_OMT_SOLVERS)
        # if preferences is not None:
        #    self.preferences.update(preferences)

    def _filter_solvers(self, solver_list, logic=None):
        """
        Returns a dict <solver_name, solver_class> including all and only
        the solvers directly or indirectly supporting the given logic.
        A solver supports a logic if either the given logic is
        declared in the LOGICS class field or if a logic subsuming the
        given logic is declared in the LOGICS class field.

        If logic is None, the map will contain all the known solvers
        """
        res = {}
        if logic is not None:
            for s, v in solver_list.items():
                for l in v.LOGICS:
                    if logic <= l:
                        res[s] = v
                        break
            return res
        else:
            solvers = solver_list

        return solvers

    def all_optimizers(self, logic=None):
        """
        Returns a dict <solver_name, solver_class> including all and only
        the solvers supporting optimization and directly or
        indirectly supporting the given logic.  A solver supports a
        logic if either the given logic is declared in the LOGICS
        class field or if a logic subsuming the given logic is
        declared in the LOGICS class field.

        If logic is None, the map will contain all the known solvers
        """
        try:
            from arlib.optimization.z3_optimizer import Z3NativeOptimizer, Z3SUAOptimizer, \
                Z3IncrementalOptimizer
            self._all_optimizers['z3'] = Z3NativeOptimizer
            self._all_optimizers['z3_sua'] = Z3SUAOptimizer
            self._all_optimizers['z3_incr'] = Z3IncrementalOptimizer
        except SolverAPINotFound:
            pass

        print("xxx: ", self._all_optimizers)
        return self._filter_solvers(self._all_optimizers, logic=logic)
