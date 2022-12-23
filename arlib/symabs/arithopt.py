# coding: utf-8
import z3
import itertools
from timeit import default_timer as symabs_timer
from typing import List
# import sys
from enum import Enum

from arlib.utils.z3_expr_utils import get_variables
from arlib.utils.z3_plus_smtlib_solver import Z3SolverPlus
from arlib.symabs.z3opt_util import box_optimize, optimize

"""
TODO: use utils.z3_plus_smtlib_solver to integrate third-party engine
"""


class OMTEngineType(Enum):
    LINEAR_SEARCH = 0
    BINARY_SEARCH = 1  # not implemented
    MIXED_LINEAR_BINARY_SEARCH = 2  # not implemented
    QUANTIFIED_SATISFACTION = 3  # has bugs??
    Z3OPT = 4
    OptiMathSAT = 5  # external?
    CVC5 = 6  # does it support int and real?
    PYSMT = 7  # what does pysmt support


class OMTEngine:
    def __init__(self):
        self.initialized = False
        self.formula = None
        self.compact_opt = True
        self.engine_type = OMTEngineType.Z3OPT

    def maximize_with_linear_search(self, obj: z3.ExprRef):
        """
        Linear Search based OMT
        """
        s = z3.Solver()
        s.add(self.formula)
        maximum = None
        if s.check() == z3.sat:
            maximum = s.model().eval(obj)
        while True:
            assumption = obj > maximum
            if s.check(assumption) == z3.unsat:
                break
            maximum = s.model().eval(obj)
            # print("current: ", maximum)
        return maximum

    def maximize_with_compact_linear_search(self, objs: List[z3.ExprRef]):
        """
        Linear Search for Boxed-OMT
        Essentially, this is equivalent to the algorithm from Li Yi's POPL 14
        """
        # print(objs)
        maximum_list = []
        s = z3.Solver()
        s.add(self.formula)
        if s.check() == z3.sat:
            for obj in objs:
                maximum_list.append(s.model().eval(obj))

        while True:
            assumption = z3.BoolVal(False)
            for i in range(len(objs)):
                assumption = z3.Or(assumption, objs[i] > maximum_list[i])
            if s.check(assumption) == z3.unsat:
                # stop here
                break
            cur_model = s.model()
            for i in range(len(objs)):
                value_of_ith_obj = cur_model.eval(objs[i])
                if value_of_ith_obj.as_long() >= maximum_list[i].as_long():
                    maximum_list[i] = value_of_ith_obj
            print("current: ", maximum_list)

        return maximum_list

    def opt_with_qsat(self, exp: z3.ExprRef, minimize: bool):
        """
        Quantified Satisfaction based OMT
        TODO: currently only works when exp is a variable (need to handle a term)?
        TODO: how to handle unbounded objectives? (seems not work??)
        """
        if z3.is_real(exp):
            exp_misc = z3.Real(str(exp) + "_m")
        else:
            exp_misc = z3.Int(str(exp) + "m")
        s = z3.Solver()
        new_fml = z3.substitute(self.formula, (exp, exp_misc))
        if minimize:
            qfml = z3.And(self.formula, z3.ForAll([exp_misc], z3.Implies(new_fml, exp <= exp_misc)))
        else:
            # TODO: why not working when x can be +oo????
            qfml = z3.And(self.formula, z3.ForAll([exp_misc], z3.Implies(new_fml, exp_misc <= exp)))
        s.add(qfml)
        if s.check() == z3.sat:
            tt = s.model().eval(exp)
            return tt
        else:
            print(s.to_smt2())
            print("UNSAT")

    def max_with_qe(self, exp: z3.ExprRef):
        """
        Quantifier Elimination based OMT
        """
        raise NotImplementedError

    def minimize_with_z3opt(self, obj: z3.ExprRef):
        return optimize(self.formula, obj, minimize=True)

    def maximize_with_z3opt(self, obj: z3.ExprRef):
        return optimize(self.formula, obj, minimize=False)

    def minimize_with_optimathsat(self, obj: z3.ExprRef):
        s = Z3SolverPlus()
        return s.optimize(self.formula, obj, minimize=True)

    def maximize_with_optimathsat(self, obj: z3.ExprRef):
        """
        Use OptiMathSAT (currently via pipe)
        """
        s = Z3SolverPlus()
        return s.optimize(self.formula, obj, minimize=False)

    def init_from_file(self, filename: str):
        try:
            self.formula = z3.And(z3.parse_smt2_file(filename))
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def init_from_fml(self, fml: z3.BoolRef):
        try:
            self.formula = fml
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def min_once(self, exp):
        """
        Minimize the objective exp
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            return -self.maximize_with_z3opt(-exp)
        elif self.engine_type == OMTEngineType.LINEAR_SEARCH:
            return -self.maximize_with_linear_search(-exp)
        elif self.engine_type == OMTEngineType.BINARY_SEARCH:
            raise NotImplementedError
        elif self.engine_type == OMTEngineType.MIXED_LINEAR_BINARY_SEARCH:
            raise NotImplementedError
        elif self.engine_type == OMTEngineType.QUANTIFIED_SATISFACTION:
            return self.opt_with_qsat(exp, minimize=True)
        elif self.engine_type == OMTEngineType.OptiMathSAT:
            return self.minimize_with_optimathsat(exp)
        else:
            return self.minimize_with_z3opt(exp)

    def max_once(self, exp: z3.ExprRef):
        """
        Maximize the objective exp
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            return self.maximize_with_z3opt(exp)
        elif self.engine_type == OMTEngineType.LINEAR_SEARCH:
            return self.maximize_with_linear_search(exp)
        elif self.engine_type == OMTEngineType.BINARY_SEARCH:
            raise NotImplementedError
        elif self.engine_type == OMTEngineType.MIXED_LINEAR_BINARY_SEARCH:
            raise NotImplementedError
        elif self.engine_type == OMTEngineType.QUANTIFIED_SATISFACTION:
            return self.opt_with_qsat(exp, minimize=False)
        elif self.engine_type == OMTEngineType.OptiMathSAT:
            return self.maximize_with_optimathsat(exp)
        else:
            return self.maximize_with_z3opt(exp)

    def min_max_many(self, multi_queries):
        """
        Boxed-OMT: compute the maximum AND minimum of queries in multi_queries
        """
        if self.engine_type == OMTEngineType.Z3OPT:
            min_res, max_res = box_optimize(self.formula, minimize=multi_queries, maximize=multi_queries, timeout=15000)
            cnts = []
            for i in range(len(multi_queries)):
                vmin = min_res[i]
                vmax = max_res[i]
                cnts.append(z3.And(multi_queries[i] >= vmin, multi_queries[i] <= vmax))
            return z3.And(cnts)
        elif self.engine_type == OMTEngineType.LINEAR_SEARCH:
            min_max_queries = []
            cnts = []
            for q in multi_queries:
                min_max_queries.append(-q)
                min_max_queries.append(q)
            res_list = self.maximize_with_compact_linear_search(min_max_queries)
            for i in range(int(len(min_max_queries) / 2)):
                cnts.append(z3.And(multi_queries[i] >= -res_list[i], multi_queries[i] <= res_list[i + 1]))
            return z3.And(cnts)
        elif self.engine_type == OMTEngineType.OptiMathSAT:
            """
            Use OptiMathSAT (currently via pipe)
            TODO: the return signature of x.box_optimize is different with the previous box...
            """
            s = Z3SolverPlus()
            return s.compute_min_max(self.formula, minimize=multi_queries, maximize=multi_queries)
        else:
            raise NotImplementedError


class NumericalAbstraction:
    """
    Symbolic Abstraction over QF_LIA and QF_LRA
    """

    def __init__(self):
        self.initialized = False
        self.formula = None
        self.vars = []
        self.omt_engine = OMTEngine()

        self.interval_abs_as_fml = None
        self.zone_abs_as_fml = None
        self.octagon_abs_as_fml = None

    def set_omt_engine_type(self, ty):
        self.omt_engine.engine_type = ty

    def do_simplification(self):
        if self.initialized:
            simp_start = symabs_timer()
            tac = z3.Then(z3.Tactic("simplify"), z3.Tactic("propagate-values"))
            simp_formula = tac.apply(self.formula).as_expr()
            simp_end = symabs_timer()
            if simp_end - simp_start > 6:
                print("error: simplification takes more than 6 seconds!!!")
            self.formula = simp_formula
        else:
            print("error: not initialized")

    def init_from_file(self, filename: str):
        try:
            self.formula = z3.And(z3.parse_smt2_file(filename))
            # NOTE: get_variables can be very flow (maybe use solver to get the var?)
            for var in get_variables(self.formula):
                if z3.is_int(var) or z3.is_real(var): self.vars.append(var)

            self.omt_engine.init_from_file(filename)
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def init_from_fml(self, fml: z3.BoolRef):
        try:
            self.formula = fml
            for var in get_variables(self.formula):
                if z3.is_int(var) or z3.is_real(var): self.vars.append(var)
            self.initialized = True

            self.omt_engine.init_from_fml(fml)

        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def to_omt_file(self, abs_type: str):
        """
        Write to OMT file
        """
        s = z3.Solver()
        s.add(self.formula)
        omt_str = s.to_smt2()
        if abs_type == "interval":
            return omt_str
        elif abs_type == "zone":
            return omt_str
        elif abs_type == "octagon":
            return omt_str
        else:
            return omt_str

    def interval_abs(self):
        if self.omt_engine.compact_opt:
            multi_queries = []
            for var in self.vars:
                multi_queries.append(var)
            self.interval_abs_as_fml = z3.simplify(self.omt_engine.min_max_many(multi_queries))
        else:
            cnts = []
            for i in range(len(self.vars)):
                vmin = self.omt_engine.min_once(self.vars[i])
                vmax = self.omt_engine.max_once((self.vars[i]))
                # print(self.vars[i], "[", vmin, ", ", vmax, "]")
                if self.omt_engine.engine_type == OMTEngineType.OptiMathSAT:
                    # TODO: this is not elegant (OptiMathSAT already returns an assertion)
                    cnts.append(z3.And(vmin, vmax))
                else:
                    cnts.append(z3.And(self.vars[i] >= vmin, self.vars[i] <= vmax))

            self.interval_abs_as_fml = z3.simplify(z3.And(cnts))
        # return simplify(And(cnts))

    def zone_abs(self):
        zones = list(itertools.combinations(self.vars, 2))
        if self.omt_engine.compact_opt:
            multi_queries = []
            for v1, v2 in zones:
                multi_queries.append(v1 - v2)

            self.zone_abs_as_fml = z3.simplify(self.omt_engine.min_max_many(multi_queries))
        else:
            zone_cnts = []
            objs = []
            for v1, v2 in zones:
                objs.append(v1 - v2)
            for exp in objs:
                exmin = self.omt_engine.min_once(exp)
                exmax = self.omt_engine.max_once(exp)
                # TODO: this is not elegant (OptiMathSAT already returns an assertion)
                if self.omt_engine.engine_type == OMTEngineType.OptiMathSAT:
                    zone_cnts.append(z3.And(exmin, exmax))
                else:
                    zone_cnts.append(z3.And(exp >= exmin, exp <= exmax))
            self.zone_abs_as_fml = z3.simplify(z3.And(zone_cnts))
        # return simplify(And(zone_cnts))

    def octagon_abs(self):
        octagons = list(itertools.combinations(self.vars, 2))
        if self.omt_engine.compact_opt:
            multi_queries = []
            for v1, v2 in octagons:
                multi_queries.append(v1 - v2)
                multi_queries.append(v1 + v2)

            self.octagon_abs_as_fml = z3.simplify(self.omt_engine.min_max_many(multi_queries))
        else:
            oct_cnts = []
            objs = []
            for v1, v2 in octagons:
                objs.append(v1 - v2)
                objs.append(v1 + v2)

            for exp in objs:
                exmin = self.omt_engine.min_once(exp)
                exmax = self.omt_engine.max_once(exp)
                # TODO: this is not elegant (OptiMathSAT already returns an assertion)
                if self.omt_engine.engine_type == OMTEngineType.OptiMathSAT:
                    oct_cnts.append(z3.And(exmin, exmax))
                else:
                    oct_cnts.append(z3.And(exp >= exmin, exp <= exmax))

            self.octagon_abs_as_fml = z3.simplify(z3.And(oct_cnts))
            # return simplify(And(oct_cnts))


def feat_test_counting():
    x, y, z = z3.Ints("x y z")
    # fml = And(x > 0, x < 1000000000000)
    fml = x > 0

    t = optimize(fml, x, minimize=False)
    s = z3.Solver()
    s.add(t > y)
    # print(s.check())
    # print(s.to_smt2())
    # exit(0)

    sa = NumericalAbstraction()
    sa.init_from_fml(fml)
    # sa.do_simplification()

    sa.set_omt_engine_type(OMTEngineType.Z3OPT)
    # sa.set_omt_engine_type(OMTEngineType.LINEAR_SEARCH)
    # sa.set_omt_engine_type(OMTEngineType.QUANTIFIED_SATISFACTION)

    sa.interval_abs()
    sa.zone_abs()
    sa.octagon_abs()
