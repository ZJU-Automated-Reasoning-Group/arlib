import sys
import subprocess
from os.path import join as path_join

UNKNOWN_TEST_CASES = [
    ("QF_UF", "test0.smt2", "model0.0.smt2"),
    ("QF_UF", "test2.smt2", "model2.0.smt2"),
    ("QF_UF", "test2.smt2", "model2.3.smt2"),
    ("QF_UF", "test2.smt2", "model2.4.smt2"),
    ("QF_UF", "test2.smt2", "model2.8.smt2"),
    ("QF_BV", "test0.smt2", "model0.empty.smt2"),
    ("QF_BV", "test1.bool.smt2", "model1.bool.smt2"),
    ("QF_BV", "test2.smt2", "model2.smt2"),
    ("QF_BV", "test2.smt2", "model2.z3.smt2"),
    ("QF_BV", "test3.smt2", "model3.z3.smt2"),
    ("QF_BV", "test4.smt2", "model4.unknown.smt2"),
    ("QF_BV", "test4.smt2", "model4.smt2"),
    ("QF_BV", "test4.smt2", "model4.malformed.smt2"),
    ("QF_BV", "test6.smt2", "model4.unknown.smt2"),
    ("QF_BV", "test6.smt2", "model0.empty.smt2"),
    ("QF_BV", "test4.smt2", "model4.sat-no-model.smt2"),
    ("QF_LIA", "test2.smt2", "model2.invalid.smt2")
]

VALID_TEST_CASES = [
    ("QF_UF", "test0.smt2", "model0.1.smt2"),
    ("QF_UF", "test0.smt2", "model0.2.smt2"),
    ("QF_UF", "test0.smt2", "model0-z3.smt2"),
    ("QF_UF", "test0.smt2", "model0-smtinterpol.smt2"),
    ("QF_UF", "test1.smt2", "model1-z3.smt2"),
    ("QF_UF", "test1.smt2", "model1-smtinterpol.smt2"),
    ("QF_UF", "test2.smt2", "model2.1.smt2"),
    ("QF_UF", "test2.smt2", "model2.2.smt2"),
    ("QF_UF", "test2.smt2", "model2.5.smt2"),
    ("QF_UF", "test2.smt2", "model2.6.smt2"),
    ("QF_UF", "test2.smt2", "model2.7.smt2"),
    ("QF_BV", "test1.smt2", "model1.cvc4.smt2"),
    ("QF_BV", "test1.smt2", "model1.z3.smt2"),
    ("QF_BV", "test1let.smt2", "model1.cvc4.smt2"),
    ("QF_BV", "test1let.smt2", "model1.z3.smt2"),
    ("QF_BV", "test2.smt2", "model2.cvc4.smt2"),
    ("QF_BV", "test3.smt2", "model3.cvc4.smt2"),
    ("QF_BV", "test6.smt2", "model6.smt2"),
    ("QF_BV", "test7.smt2", "model7.smt2"),
    ("QF_BV", "test8.smt2", "model8.smt2"),
    ("QF_BV", "test8.smt2", "model8.2.smt2"),
    ("QF_BV", "test10.smt2", "model10.smt2"),
    ("QF_BV", "test9.smt2", "model9.cvc4.smt2"),
    ("QF_LIA", "test1.smt2", "model1.smtinterpol.smt2"),
    ("QF_LIA", "test1.smt2", "model1.z3.smt2"),
    ("QF_LIA", "test1.smt2", "model1.cvc4.smt2"),
    ("QF_LIA", "test2.smt2", "model2.smtinterpol.smt2"),
    ("QF_LIA", "test2.smt2", "model2.z3.smt2"),
    ("QF_LIA", "test2.smt2", "model2.cvc4.smt2"),
    ("QF_LRA", "test0.smt2", "model0.1.smt2"),
    ("QF_LRA", "test0.smt2", "model0.2.smt2"),
    ("QF_LRA", "test0.smt2", "model0.3.smt2"),
    ("QF_LRA", "test0.smt2", "model0.5.smt2"),
    ("QF_LRA", "test0.smt2", "model0.6.smt2"),
    ("QF_LRA", "test0.smt2", "model0.7.smt2"),
    ("QF_LRA", "test3.smt2", "model3.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.2.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.3.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.4.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.5.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.6.smt2"),
    ("QF_LIRA", "test0.smt2", "model0.7.smt2"),
    ("QF_LIRA", "test1.smt2", "model1.smtinterpol.smt2"),
    ("QF_LIRA", "test1.smt2", "model1.cvc4.smt2"),
    ("QF_LIRA", "test3.smt2", "model3.smt2"),
    ("QF_LIRA", "LCTES_smtopt.smt2", "model.LCTES_smtopt.cvc4.smt2"),
    ("QF_LIRA", "LCTES_smtopt.smt2", "model.LCTES_smtopt.z3.smt2"),
    ("QF_LIRA", "LCTES_smtopt.smt2", "model.LCTES_smtopt.smtinterpol.smt2"),
]

INVALID_TEST_CASES = [
    ("QF_BV", "test5.smt2", "model5.cvc4.smt2"),
    ("QF_BV", "test5.smt2", "model5.z3.smt2"),
    ("QF_BV", "test5.smt2", "model6.unsat.smt2"),
    ("QF_BV", "test6.smt2", "model6.unsat.smt2"),
    ("QF_BV", "test6.smt2", "model5.cvc4.smt2"),
    ("QF_BV", "test6.smt2", "model5.z3.smt2"),
    ("QF_BV", "test9.smt2", "model9.z3.smt2"),
    ("QF_LIRA", "test1.smt2", "model1.broken.smt2"),
]

BASE_DIR = "examples/"


def validate(problem, model, result):
    interpreter = sys.executable
    args = [interpreter, "smt_model_validator.py",
            "--smt2", problem,
            "--model", model]
    print(args)
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    msg = res.stdout.decode('ascii')
    return msg.startswith("model_validator_status=" + result), msg


for division, problem, model in UNKNOWN_TEST_CASES:
    print("testing UNKNOWN {} problem {} with model {}...".format(division, problem, model))
    res, msg = validate(path_join(BASE_DIR, division, problem),
                        path_join(BASE_DIR, division, model), "UNKNOWN")
    assert res, (division, problem, model, msg)
    print("OK")

for division, problem, model in VALID_TEST_CASES:
    print("testing VALID {} problem {} with model {}...".format(division, problem, model))
    res, msg = validate(path_join(BASE_DIR, division, problem),
                        path_join(BASE_DIR, division, model), "VALID")
    assert res, (division, problem, model, msg)
    print("OK")

for division, problem, model in INVALID_TEST_CASES:
    print("testing INVALID {} problem {} with model {}...".format(division, problem, model))
    res, msg = validate(path_join(BASE_DIR, division, problem),
                        path_join(BASE_DIR, division, model), "INVALID")
    if res:
        print("OK")
    else:
        print("FAILED!")
        print((division, problem, model, msg))
