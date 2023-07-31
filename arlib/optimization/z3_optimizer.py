"""
OMT based on Z3
"""

from __future__ import absolute_import

from pysmt.solvers.z3 import Z3Solver, Z3Model

from arlib.optimization.omt_exceptions import OMTInfinityError, SolverAPINotFound, \
    OMTUnboundedOptimizationError, OMTInfinitesimalError, GoalNotSupportedError

from arlib.optimization.optimizer import Optimizer

try:
    import z3
except ImportError:
    raise SolverAPINotFound


class Z3NativeOptimizer(Optimizer, Z3Solver):
    """OMT based on Z3"""
    LOGICS = Z3Solver.LOGICS

    def __init__(self, environment, logic, **options):
        Z3Solver.__init__(self, environment=environment,
                          logic=logic, **options)
        self.z3 = z3.Optimize()

    def _assert_z3_goal(self, goal):
        h = None
        if goal.is_maxsmt_goal():
            for soft, w in goal.soft:
                obj_soft = self.converter.convert(soft)
                h = self.z3.add_soft(obj_soft, w, "__pysmt_" + str(goal.id))
        else:
            term = goal.term()
            ty = self.environment.stc.get_type(term)
            if goal.signed and ty.is_bv_type():
                width = ty.width
                term = self.mgr.BVAdd(term, self.mgr.BV(2 ** (width - 1), width))
            obj = self.converter.convert(term)
            if goal.is_minimization_goal():
                h = self.z3.minimize(obj)
            elif goal.is_maximization_goal():
                h = self.z3.maximize(obj)
            else:
                raise GoalNotSupportedError("z3", goal.__class__)
        return h

    def optimize(self, goal, **kwargs):
        h = self._assert_z3_goal(goal)

        res = self.z3.check()
        if res == z3.sat:
            try:
                if goal.is_maxsmt_goal():
                    model = Z3Model(self.environment, self.z3.model())
                    return model, None
                else:
                    opt_value = self.z3.lower(h)
                    self.converter.back(opt_value)
                    model = Z3Model(self.environment, self.z3.model())
                    return model, model.get_value(goal.term())
            except OMTInfinityError:
                raise OMTUnboundedOptimizationError("The optimal value is unbounded")
            except OMTInfinitesimalError:
                raise OMTUnboundedOptimizationError("The optimal value is infinitesimal")
        else:
            return None

    def pareto_optimize(self, goals):
        self.z3.set(priority='pareto')
        for goal in goals:
            self._assert_z3_goal(goal)
        while self.z3.check() == z3.sat:
            model = Z3Model(self.environment, self.z3.model())
            yield model, [model.get_value(x.term()) for x in goals]

    def can_diverge_for_unbounded_cases(self):
        return False

    def boxed_optimize(self, goals):
        self.z3.set(priority='box')
        models = {}
        for goal in goals:
            self._assert_z3_goal(goal)

        for goal in goals:
            if self.z3.check() == z3.sat:
                model = Z3Model(self.environment, self.z3.model())
                models[goal] = (model, model.get_value(goal.term()))
            else:
                return None

        return models

    def lexicographic_optimize(self, goals):
        self.z3.set(priority='lex')
        for goal in goals:
            self._assert_z3_goal(goal)

        if self.z3.check() == z3.sat:
            model = Z3Model(self.environment, self.z3.model())
            return model, [model.get_value(x.term()) for x in goals]
        else:
            return None, None
