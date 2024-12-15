#!/usr/bin/env python3
# coding: utf-8

import random
import z3
import itertools
from functools import reduce
from arlib.utils.z3_expr_utils import get_variables


# Approach taken from:
#   Rafael Dutra, Kevin Laeufer, Jonathan Bachrach and Koushik Sen:
#   Efficient Sampling of SAT Solutions for Testing, ICSE 2018.
#   https://github.com/RafaelTupynamba/quicksampler/

# TODO: The generated samples are currently not checked for whether they satisfy the given constraints!

# https://stackoverflow.com/questions/39299015/sum-of-all-the-bits-in-a-bit-vector-of-z3
def bvcount(b):
    n = b.size()
    bits = [z3.Extract(i, i, b) for i in range(n)]
    bvs = [z3.Concat(z3.BitVecVal(0, n - 1), b) for b in bits]
    nb = reduce(lambda a, b: a + b, bvs)
    return nb


MAX_LEVEL = 6


def cast_long_to_str(x, n):
    # see angr/state_plugins/solver.py _cast_to
    # return '{:x}'.format(x).zfill(n/4).decode('hex')
    return '{:x}'.format(x).zfill(n // 4)  # .decode('hex')


def bvsampler(constraints, target):
    # targe can onl be a variable???
    n = target.size()

    solver = z3.Optimize()
    solver.add(constraints)
    delta = z3.BitVec('delta', n)
    result = z3.BitVec('result', n)
    solver.add(result == target)
    solver.minimize(bvcount(delta))

    results = set()

    while True:
        # print('---------------------------')
        guess = z3.BitVecVal(random.getrandbits(n), n)

        solver.push()
        solver.add(result ^ delta == guess)

        if solver.check() != z3.sat:
            break

        model = solver.model()
        result0 = model[result].as_long()
        solver.pop()

        results.add(result0)
        yield result0

        # print('solver: ' + str(solver))
        # print('guess: ' + str(guess))
        # print('model: ' + str(model))

        mutations = {}

        solver.push()

        for i in range(n):
            # print('mutating bit ' + str(i))
            solver.push()
            goal = z3.BitVecVal(result0, n)
            solver.add(result ^ delta == goal)
            solver.add(z3.Extract(i, i, delta) == 0x1)

            if solver.check() == z3.sat:
                model = solver.model()
                result1 = model[result].as_long()

                if result1 not in results:
                    results.add(result1)
                    yield result1

                new_mutations = {}
                new_mutations[result1] = 1

                for value in mutations:
                    level = mutations[value]
                    if level > MAX_LEVEL:
                        continue

                    candidate = (result0 ^ ((result0 ^ value) | (result0 ^ result1)))
                    # print('yielding candidate ' + str(candidate) + ' at level ' + str(level))
                    if candidate not in results:
                        results.add(candidate)
                        yield candidate

                    new_mutations[candidate] = level + 1

                mutations.update(new_mutations)

            solver.pop()

        solver.pop()


def test_sampler():
    x = z3.BitVec('x', 16)
    y = z3.BitVec('y', 16)
    # sample = bvsampler(z3.And(x > 1000, x < 10000, z3.Or(x < 4000, x > 5000)), x)
    sample = bvsampler(z3.And(x > 1000, y < 10000, z3.Or(x < 4000, x > 5000)), x)
    print("Hello")

    for x in sample:
        y = cast_long_to_str(x, 16)
        print('possible solution: ' + y)


def quicksampler_for_file(fname):
    try:
        fvec = z3.parse_smt2_file(fname)
        formula = z3.And(fvec)
        vars = get_variables(formula)
        print("start")
        sample = bvsampler(formula, vars[0])
        for x in sample:
            y = cast_long_to_str(x, 16)
            print('possible solution: ' + y)

    except z3.Z3Exception as e:
        print(e)

    return


if __name__ == '__main__':
    # test_sampler()
    quicksampler_for_file('../test/t1.smt2')
