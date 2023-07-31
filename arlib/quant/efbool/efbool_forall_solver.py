import logging
from typing import List

from pysat.formula import CNF
from pysat.solvers import Solver

logger = logging.getLogger(__name__)


class BoolForAllSolver(object):
    def __init__(self, exists_vars: List[int], forall_vars: List[int], clauses: List[List[int]]):
        self.solver_name = "m22"
        self.reduce_model = False
        pos = CNF(from_clauses=clauses)
        negated = pos.negate()
        self.neg_clauses = negated.clauses
        self.existential_bools = exists_vars
        self.univeral_bools = forall_vars

    def reduce_counter_example(self, existential_model: List, existential_counter_model: List) -> List:
        """http://fmv.jku.at/papers/NiemetzPreinerBiere-FMCAD14.pdf
        Consider a Boolean formula P. The model of P (given by a SAT solver) is not necessarily minimal.
        In other words, the SAT solver may assign truth assignments to literals irrelevant to truth of P.

        Suppose we have a model M of P. To extract a smaller assignment, one trick is to encode the
        negation of P in a separate dual SAT solver.

        We can pass M as an assumption to the dual SAT solver. (check-sat-assuming M).
        All assumptions inconsistent with -P (called the failed assumptions),
        are input assignments sufficient to falsify -P, hence sufficient to satisfy P.

        Related work
          - https://arxiv.org/pdf/2110.12924.pdf
        """
        pos = CNF(from_clauses=self.neg_clauses)
        pos.append(existential_model)
        neg = pos.negate()

        aux_sol = Solver(name="m22", bootstrap_with=neg)
        assert not aux_sol.solve(assumptions=existential_counter_model)
        return aux_sol.get_core()

    def check_models(self, models: List[List[int]]):
        """ Check candidates given by the exists solver
        """
        blocking_clauses = []
        for existential_model in models:
            # TODO: check them in parallel?
            solver = Solver(name="m22", bootstrap_with=self.neg_clauses)
            for v in existential_model:
                solver.add_clause([v])
            # NOTE: should not directly pass existential_model
            ans = solver.solve()
            if not ans:
                # at least one existential model is good
                logger.debug("f-solver success?")
                logger.debug(existential_model)
                return []
            # let e_model be  [-4, -7, 6]. the blocking clauses should be [4, 7, -6]
            # FIXME: why not using the counterexample from self.solver?
            if self.reduce_model:
                existential_counter_model = []
                for val in solver.get_model():
                    if abs(val) in self.existential_bools:
                        existential_counter_model.append(val)
                blocking_clauses.append(self.reduce_counter_example(existential_model, existential_counter_model))
            else:
                blocking_clauses.append([-v for v in existential_model])

        return blocking_clauses
