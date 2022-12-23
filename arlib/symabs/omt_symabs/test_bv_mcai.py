# coding: utf-8
import z3

from arlib.tests.formula_generator import FormulaGenerator
from arlib.symabs.omt_symabs.bv_symbolic_abstraction import BVSymbolicAbstraction
from arlib.bv.bvcounting import ModelCounter
# from ..utils.plot_util import ScatterPlot  # See arlib/scripts


def is_sat(e):
    s = z3.Solver()
    s.add(e)
    s.set("timeout", 6000)
    return s.check() == z3.sat


def model_count(fml):
    mc = ModelCounter()
    mc.init_from_fml(fml)
    return mc.count_models_by_sharp_sat()


def has_fp(abs_formula, fml):
    solver = z3.Solver()
    solver.add(z3.And(abs_formula, z3.Not(fml)))
    if solver.check() == z3.unsat:
        # print("abs has no false positives!")
        return False
    return True


def compute_fp_rate(abs_formula, fml):
    mc_abs = ModelCounter()
    mc_abs.init_from_fml(abs_formula)
    abs_count = mc_abs.count_models_by_sharp_sat()
    # print("Model count of the abstraction: ", abs_count)

    # Count the number of false positives
    fp_fml = z3.And(abs_formula, z3.Not(fml))
    mc_abs_fp = ModelCounter()
    mc_abs_fp.init_from_fml(fp_fml)
    abs_fp_count = mc_abs_fp.count_models_by_sharp_sat()
    # print("Model count of false positives: ", abs_fp_count)

    return abs_fp_count / abs_count


interval_fps = []
zone_fps = []
octaogn_fps = []


def test_mcai():
    global interval_fps, zone_fps, octaogn_fps
    """
    TODO: handle timeout; log time and results
    """
    try:
        v, w, x, y, z = z3.BitVecs("v w x y z", 4)
        fg = FormulaGenerator([v, w, x, y, z])
        fml = fg.generate_formula()
        if is_sat(fml):
            # 1. Do abstraction
            sa = BVSymbolicAbstraction()
            sa.init_from_fml(fml)
            # sa.do_simplification()

            sa.interval_abs()
            sa.zone_abs()
            sa.octagon_abs()

            ret = False

            # 2. Count false positives
            if has_fp(sa.interval_abs_as_fml, fml):
                fpr_int = compute_fp_rate(sa.interval_abs_as_fml, fml)
                print("Interval domain FP rate: ", fpr_int)
            else:
                fpr_int = 0
                print("Interval domain no FP")
            interval_fps.append(fpr_int)

            if has_fp(sa.zone_abs_as_fml, fml):
                fpr_zone = compute_fp_rate(sa.zone_abs_as_fml, fml)
                print("Zone domain FP rate: ", fpr_zone)
            else:
                fpr_zone = 0
                print("Zone domain no FP")
            zone_fps.append(fpr_zone)

            if has_fp(sa.octagon_abs_as_fml, fml):
                fpr_oct = compute_fp_rate(sa.octagon_abs_as_fml, fml)
                print("Octagon domain FP rate: ", fpr_oct)
            else:
                fpr_oct = 0
                print("Octagon domain no FP")
            octaogn_fps.append(fpr_oct)

            print("")
            """
            if fpr_zone == fpr_int:
                print(sa.interval_abs_as_fml)
                print("-------------------")
                print(sa.zone_abs_as_fml)
            """
            return ret

        return False
    except Exception as ex:
        print(ex)
        return False


"""
# NOTE: ths needs matplotlib (See arlib/scripts/plot.py)
def test_mcai_plot():
    for _ in range(100):
        # if test():
        if test_mcai():
            break
    pp = ScatterPlot(name_a="interval", name_b="octagon")
    pp.get_scatter_plot((interval_fps, octaogn_fps))
"""


def main():
    try:
        x, y, z = z3.BitVecs("x y z", 8)
        fg = FormulaGenerator([x, y, z])
        fml = fg.generate_formula()
        if is_sat(fml):
            # print(fml)
            mc = ModelCounter()
            mc.init_from_fml(fml)
            # bit_count_res = mc.count_models_by_bits_enumeration()
            # print("bit-level enumeration result: ", bit_count_res)
            sharp_sat_res = mc.count_models_by_sharp_sat()
            print("sharpSAT result: ", sharp_sat_res)
            # mc.count_model_by_bv_enumeration() # why different with the previous one (due to overflow? bit-blasting?
            return True
        return False
    except Exception as ex:
        print(ex)
        return False


if __name__ == '__main__':
    main()
