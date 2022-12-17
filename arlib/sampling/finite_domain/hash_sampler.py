# Enumerate models with some random parity constraints as suggested for approximate model counting.
# Taken from https://github.com/Z3Prover/z3/issues/4675#issuecomment-686880139
from z3 import *
from random import *


def get_uniform_samples_with_xor(vars, cnt, num_samples):
    """
    Get num_samples models (projected to vars)
    """
    res = []
    s = Solver()
    s.add(cnt)
    bits = []
    for var in vars:
        bits = bits + [Extract(i, i, var) == 1 for i in range(var.size())]
    num_success = 0
    while True:
        s.push()
        rounds = 3  # why 3?
        for x in range(rounds):
            trials = 10
            fml = BoolVal(randrange(0, 2))
            for i in range(trials):
                fml = Xor(fml, bits[randrange(0, len(bits))])
            s.add(fml)
        if s.check() == sat:
            res.append([s.model().eval(var, True) for var in vars])
            num_success += 1
            if num_success == num_samples:
                break
        s.pop()
    return res


def test_api():
    x, y, z = BitVecs('x y z', 32)
    fml = And(ULT(x, 13), ULT(y, x), ULE(y, z))
    print(get_uniform_samples_with_xor([x, y], fml, 8))


def test(num_samples):
    x, y = BitVecs('x y', 48)
    s = Solver()
    s.add(ULT(x, 13))
    s.add(ULT(y, x))
    bits = [Extract(i, i, x) == 1 for i in range(48)] + [Extract(i, i, y) == 1 for i in range(48)]
    rand = random()
    num_success = 0
    while True:
        s.push()
        rounds = 3  # why 3?
        for x in range(rounds):
            trials = 10
            fml = BoolVal(randrange(0, 2))
            for i in range(trials):
                fml = Xor(fml, bits[randrange(0, len(bits))])
            s.add(fml)
        if s.check() == sat:
            num_success += 1
            if num_success == num_samples:
                break
            print(s.model())
        s.pop()


# test(8)

test_api()
