"""
This module provides a symbolic abstraction for bit-vector formulas.
It supports interval, zone, and octagon abstractions.
"""

import z3
import itertools
from timeit import default_timer as symabs_timer
from typing import List

from arlib.utils.z3_expr_utils import get_variables
from arlib.utils.z3_solver_utils import is_entail
from arlib.symabs.omt_symabs.z3opt_util import box_optimize


# import argparse


def get_bv_size(x: z3.ExprRef):
    if z3.is_bv(x):
        return x.sort().size()
    else:
        return -1


class BVSymbolicAbstraction:
    def __init__(self):
        self.initialized = False
        self.formula = z3.BoolVal(True)
        self.vars = []
        self.bool_vars = []
        self.interval_abs_as_fml = z3.BoolVal(True)
        self.zone_abs_as_fml = z3.BoolVal(True)
        self.octagon_abs_as_fml = z3.BoolVal(True)
        self.bitwise_abs_as_fml = z3.BoolVal(True)

        self.single_query_timeout = 5000
        self.multi_query_tiemout = 0
        # self.poly_abs_as_fml = BoolVal(True)

        self.compact_opt = True

        self.obj_no_overflow = False
        self.obj_no_underflow = False

        self.signed = False
        # set_param("verbose", 15)

    def do_simplification(self):
        """
        Simplify the formula using Z3 tactics.
        If the formula is initialized, apply a sequence of tactics to simplify it.
        """
        if self.initialized:
            # TODO: it seems that propagate-bv-bounds has bugs, which can be even non-terminating on some formulas
            # TODO: use try_for?
            simp_start = symabs_timer()
            tac = z3.Then(z3.Tactic("simplify"), z3.Tactic("propagate-values"), z3.Tactic("propagate-bv-bounds"))
            simp_formula = tac.apply(self.formula).as_expr()
            simp_end = symabs_timer()
            if simp_end - simp_start > 6:
                print("error: simp takes more than 6 seconds!!!")
            self.formula = simp_formula
        else:
            print("error: not initialized")

    def init_from_file(self, fname: str):
        try:
            fvec = z3.parse_smt2_file(fname)
            self.formula = z3.And(fvec)
            # NOTE: get_variables can be very flow (maybe use solver to get the var?)
            all_vars = get_variables(self.formula)
            for var in all_vars:
                if z3.is_bv(var):
                    self.vars.append(var)
                elif z3.is_bool(var):
                    self.bool_vars.append(var)
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def init_from_fml(self, fml: z3.BoolRef):
        try:
            self.formula = fml
            for var in get_variables(self.formula):
                if z3.is_bv(var):
                    self.vars.append(var)
                elif z3.is_bool(var):
                    self.bool_vars.append(var)
            self.initialized = True
        except z3.Z3Exception as ex:
            print("error when initialization")
            print(ex)

    def min_once(self, exp: z3.ExprRef):
        """
        Minimize exp
        """
        sol = z3.Optimize()
        sol.set("timeout", self.single_query_timeout)
        sol.add(self.formula)
        sol.minimize(exp)
        if sol.check() == z3.sat:
            m = sol.model()
            return m.eval(exp, True)
            # return m.eval(exp).as_long()

    def max_once(self, exp: z3.ExprRef):
        sol = z3.Optimize()
        sol.set("timeout", self.single_query_timeout)
        sol.add(self.formula)
        sol.maximize(exp)
        if sol.check() == z3.sat:
            m = sol.model()
            # print(m)
            return m.eval(exp, True)
            # return m.eval(exp).as_long()

    def min_max_many(self, multi_queries: List[z3.ExprRef]):
        """
        Minimize and maximize the given multi_queries.
        Returns the conjunction of the minimized and maximized expressions.
        """

        # n_queries = len(multi_queries)
        # timeout = n_queries * self.single_query_timeout * 2 # is this reasonable?
        min_res, max_res = box_optimize(self.formula, minimize=multi_queries, maximize=multi_queries)
        # TODO: the res of handler.xx() is not a BitVec val, but Int?
        # TODO: what if it is a value large than the biggest integer of the size (is it possible? e.g., due to overflow)
        cnts = []
        for i in range(len(multi_queries)):
            vmin = min_res[i]
            vmax = max_res[i]
            vmin_bvval = z3.BitVecVal(vmin.as_long(), multi_queries[i].sort().size())
            vmax_bvval = z3.BitVecVal(vmax.as_long(), multi_queries[i].sort().size())
            # print(self.vars[i].sort(), vmin.sort(), vmax.sort())
            if self.signed:
                cnts.append(z3.And(multi_queries[i] >= vmin_bvval, multi_queries[i] <= vmax_bvval))
            else:
                cnts.append(z3.And(z3.UGE(multi_queries[i], vmin_bvval), z3.ULE(multi_queries[i], vmax_bvval)))
        return z3.And(cnts)

    def interval_abs(self):
        """
        Perform interval abstraction on the formula.
        Compute the minimum and maximum values for each variable in the formula.
        Store the result in self.interval_abs_as_fml.
        """
        if self.compact_opt:
            # Use multi-query optimization for better performance
            multi_queries = [var for var in self.vars]
            self.interval_abs_as_fml = self.min_max_many(multi_queries)
        else:
            # Compute intervals one variable at a time
            constraints = []
            for var in self.vars:
                vmin = self.min_once(var)
                vmax = self.max_once(var)
                
                # Add bounds based on signed/unsigned comparison
                if self.signed:
                    constraint = z3.And(var >= vmin, var <= vmax)
                else:
                    constraint = z3.And(z3.UGE(var, vmin), z3.ULE(var, vmax))
                constraints.append(constraint)
            self.interval_abs_as_fml = z3.And(constraints)
        self.interval_abs_as_fml = z3.And(self.interval_abs_as_fml, z3.And([bool_var == bool_var for bool_var in self.bool_vars]))
        print("\ninterval abs:", self.interval_abs_as_fml, sep="\n")

    def zone_abs(self):
        """
        Perform zone abstraction on the formula.
        Compute the minimum and maximum values for each pair of variables in the formula.
        Store the result in self.zone_abs_as_fml.
        """
        zones = list(itertools.combinations(self.vars, 2))
        tmp = self.formula
        if self.compact_opt:
            multi_queries = []
            wrap_around_cnts = []

            for var in self.vars:
                multi_queries.append(var)

            for v1, v2 in zones:
                if v1.sort().size() == v2.sort().size():
                    multi_queries.append(v1 - v2)
                    if self.obj_no_overflow:
                        wrap_around_cnts.append(z3.BVSubNoOverflow(v1, v2))
                    if self.obj_no_underflow:
                        wrap_around_cnts.append(z3.BVSubNoUnderflow(v1, v2, signed=self.signed))

            if len(wrap_around_cnts) > 1:
                self.formula = z3.And(self.formula, z3.And(wrap_around_cnts))

            self.zone_abs_as_fml = self.min_max_many(multi_queries)
        else:
            zone_cnts = []
            objs = []
            wrap_around_cnts = []

            for var in self.vars:
                objs.append(var)

            for v1, v2 in zones:
                if v1.sort().size() == v2.sort().size():
                    objs.append(v1 - v2)
                    if self.obj_no_overflow:
                        wrap_around_cnts.append(z3.BVSubNoOverflow(v1, v2))
                    if self.obj_no_underflow:
                        wrap_around_cnts.append(z3.BVSubNoUnderflow(v1, v2, signed=self.signed))

            if len(wrap_around_cnts) > 1:
                self.formula = z3.And(self.formula, z3.And(wrap_around_cnts))

            for exp in objs:
                # TODO: use BVSubNoOverflow
                exmin = self.min_once(exp)
                exmax = self.max_once(exp)
                if self.signed:
                    zone_cnts.append(z3.And(exp >= exmin, exp <= exmax))
                else:
                    zone_cnts.append(z3.And(z3.UGE(exp, exmin), z3.ULE(exp, exmax)))

            self.zone_abs_as_fml = z3.And(zone_cnts)
        self.zone_abs_as_fml = z3.And(self.zone_abs_as_fml, z3.And([bool_var == bool_var for bool_var in self.bool_vars]))
        self.formula = tmp
        print("\nzone abs:", self.zone_abs_as_fml, sep="\n")

    def octagon_abs(self):
        """
        Perform octagon abstraction on the formula.
        Compute the minimum and maximum values for each pair of variables in the formula,
        Store the result in self.octagon_abs_as_fml.
        """
        octagons = list(itertools.combinations(self.vars, 2))
        tmp = self.formula
        if self.compact_opt:
            multi_queries = []
            wrap_around_cnts = []

            for var in self.vars:
                # need this?
                multi_queries.append(var)

            for v1, v2 in octagons:
                if v1.sort().size() == v2.sort().size():
                    multi_queries.append(v1 - v2)
                    multi_queries.append(v1 + v2)
                    if self.obj_no_overflow:
                        wrap_around_cnts.append(z3.BVSubNoOverflow(v1, v2))
                        wrap_around_cnts.append(z3.BVAddNoOverflow(v1, v2, signed=self.signed))
                    if self.obj_no_underflow:
                        wrap_around_cnts.append(z3.BVSubNoUnderflow(v1, v2, signed=self.signed))
                        wrap_around_cnts.append(z3.BVAddNoUnderflow(v1, v2))

            if len(wrap_around_cnts) > 1:
                self.formula = z3.And(self.formula, z3.And(wrap_around_cnts))

            self.octagon_abs_as_fml = self.min_max_many(multi_queries)
        else:
            oct_cnts = []
            objs = []
            wrap_around_cnts = []

            for var in self.vars:
                # need this?
                objs.append(var)

            for v1, v2 in octagons:
                if v1.sort().size() == v2.sort().size():
                    objs.append(v1 - v2)
                    objs.append(v1 + v2)
                    if self.obj_no_overflow:
                        wrap_around_cnts.append(z3.BVSubNoOverflow(v1, v2))
                        wrap_around_cnts.append(z3.BVAddNoOverflow(v1, v2, signed=self.signed))
                    if self.obj_no_underflow:
                        wrap_around_cnts.append(z3.BVSubNoUnderflow(v1, v2, signed=self.signed))
                        wrap_around_cnts.append(z3.BVAddNoUnderflow(v1, v2))

            if len(wrap_around_cnts) > 1:
                self.formula = z3.And(self.formula, z3.And(wrap_around_cnts))

            for exp in objs:
                exmin = self.min_once(exp)
                exmax = self.max_once(exp)
                if self.signed:
                    oct_cnts.append(z3.And(exp >= exmin, exp <= exmax))
                else:
                    oct_cnts.append(z3.And(z3.UGE(exp, exmin), z3.ULE(exp, exmax)))

            self.octagon_abs_as_fml = z3.And(oct_cnts)
        self.octagon_abs_as_fml = z3.And(self.octagon_abs_as_fml, z3.And([bool_var == bool_var for bool_var in self.bool_vars]))
        self.formula = tmp
        print("\noctagon abs:", self.octagon_abs_as_fml, sep="\n")
    
    def bitwise_abs(self):
        """
        Perform octagon abstraction on the formula.
        extract every bit of the variables and compute if the bit must be 0 or 1.
        Store the result in self.bitwise_abs_as_fml.
        """
        cnts = []
        for var in self.vars:
            for i in range(var.size()):
                sol = z3.Solver()
                sol.add(self.formula)
                sol.push()
                sol.add(z3.Extract(i, i, var) == 0)
                if sol.check() == z3.unsat:
                    cnts.append(z3.Extract(i, i, var) == 1)
                    continue
                sol.pop()
                sol.add(z3.Extract(i, i, var) == 1)
                if sol.check() == z3.unsat:
                    cnts.append(z3.Extract(i, i, var) == 0)
                    continue
                cnts.append(z3.Or(z3.Extract(i, i, var) == 0, z3.Extract(i, i, var) == 1))
        # v = self.vars[::]
        # for var1 in self.vars:
        #     for var2 in self.vars:
        #         if var1 == var2:
        #             continue
        #         if var2 ^ var1 not in v:
        #             v.append(var1 ^ var2)
        #         if var2 & var1 not in v:
        #             v.append(var1 & var2)
        #         if var2 | var1 not in v:
        #             v.append(var1 | var2)
        # # for var in v:
        # #     flag = True
        # #     for i in range(var.size() - 1):
        # #         sol = z3.Solver()
        # #         sol.add(self.formula)
        # #         for b1 in (0, 1):
        # #             for b2 in (0, 1):
        # #                 sol.push()
        # #                 sol.add(z3.Extract(i, i, var) == b1)
        # #                 sol.add(z3.Extract(i + 1, i + 1, var) == b2)
        # #                 if sol.check() == z3.unsat:
        # #                     cnts.append(z3.Extract(i + 1, i, var) != b2 * 2 + b1)
        # #                     flag = False
        # #                 sol.pop()
        # #     if flag and var in self.vars:
        # #         cnts.append(var == var)
        # for var in v:
        #     flag = True
        #     sol = z3.Solver()
        #     sol.add(self.formula)
        #     for i in range(var.size()):
        #         for b in (0, 1):
        #             sol.push()
        #             sol.add(z3.Extract(i, i, var) == b)
        #             if sol.check() == z3.unsat:
        #                 cnts.append(z3.Extract(i, i, var) == 1 - b)
        #                 flag = False
        #             sol.pop()
        #     if flag and var in self.vars:
        #         cnts.append(var == var)                
        self.bitwise_abs_as_fml = z3.And(cnts)
        self.bitwise_abs_as_fml = z3.And(self.bitwise_abs_as_fml, z3.And([bool_var == bool_var for bool_var in self.bool_vars]))
        print("\nbitwise abs:", self.bitwise_abs_as_fml)

def feat_test():
    x = z3.BitVec("x", 8)
    y = z3.BitVec("y", 8)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x, 10), z3.ULT(y, 10))
    sa = BVSymbolicAbstraction()
    sa.init_from_fml(fml)
    sa.interval_abs()
    sa.zone_abs()


def feat_test_counting():
    x = z3.BitVec("x", 8)
    y = z3.BitVec("y", 8)
    z = z3.BitVec("z", 8)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 4), z3.ULT(x, 10), z3.ULT(z, 10))

    sa = BVSymbolicAbstraction()
    # sa.init_from_file(fname)
    sa.init_from_fml(fml)
    sa.do_simplification()

    # sa.zone_abs()
    # abs_formula = sa.zone_abs_as_fml
    # sa.octagon_abs();
    sa.interval_abs()
    abs_formula = sa.interval_abs_as_fml

    if is_entail(abs_formula, sa.formula):
        print("abs is a sound over-approximation of the orignal formula!")

    solver = z3.Solver()
    # the solution space described by the intervals
    solver.add(abs_formula)
    # negate the solution space described by the truth formula
    solver.add(z3.Not(sa.formula))
    check_res = solver.check()
    # print(solver.check())
    if check_res == z3.unsat:
        print("abs has no false positives!")
    else:
        print("abs has false positives!")
        '''
        # Count the number of false positives
        # print(solver.to_smt2()) # TODO: may save as a file
        fp_fml = And(solver.assertions())
        mc_abs = ModelCountrer()
        mc_abs.init_from_fml(fp_fml)
        # mc_abs.count_models_by_bit_enumeration()
        mc_abs.count_model_by_bv_enumeration()
        # mc_abs.count_models_by_sharpSAT()
        '''


def test():
    # test_multi_opt()
    feat_test_counting('../test/t1.smt2')


if __name__ == '__main__':
    test()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', dest='domain', default='interval', type=str,
                      help="domain: interval, octagon, zone")
    parser.add_argument('--timeout', dest='timeout', default=30, type=int, help="timeout")
    parser.add_argument('--file', dest='file', default='none', type=str, help="file")

    args = parser.parse_args()
    main(args.file, args.timeout, args.domain)
    '''
