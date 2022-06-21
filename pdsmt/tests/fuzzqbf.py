#!/usr/bin/env python3

"""
    params:
        m = number of clauses
        x = number of existentials
        u = number of universals
        d = number of deps
        w = existential clause width
        v = universal clause width
"""

import argparse
import random


def litstr(lits):
    return " ".join(map(str, lits))


def sample_subset(S, k=None):
    """
        sample a dependency set of size d out of U
        if d is not given, sample a subset uniformly
    """
    if k is not None:
        return random.sample(S, k)
    else:
        return [x for x in S if random.randint(0, 1)]


def sample_clause(U, v, X, w):
    """
        sample a clause uniformly at random with
            v variables from the set U
            w variables from the set X
        if v or w not given, sample a random susbet of variables
    """
    return frozenset(random.choice([-1, 1]) * var for var in sample_subset(U, v) + sample_subset(X, w))


def sample_param(p, s):
    return random.randint(int(p * (1 - s)), int(p * (1 + s)))


def main(args):
    u = sample_param(args.u, args.s)
    x = sample_param(args.x, args.s)
    m = sample_param(args.m, args.s)
    v = sample_param(args.v, args.s) if args.d is not None else None
    w = sample_param(args.w, args.s) if args.d is not None else None
    d = sample_param(args.d, args.s) if args.d is not None else None

    U = range(1, u + 1)
    X = range(u + 1, u + x + 1)
    S = {x: sample_subset(U, k=d) for x in X}
    F = set()
    while len(F) < m:
        F.add(sample_clause(U, v, X, w))

    print(f"p cnf {x + u} {m}")
    print("a " + litstr(U) + " 0")
    for xvar in X:
        print(f"d {xvar} " + litstr(S[xvar]) + " 0")
    for C in F:
        print(litstr(C) + " 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=50, help="number of clauses")
    parser.add_argument("-w", type=int, default=None, help="existential clause width")
    parser.add_argument("-v", type=int, default=None, help="universal clause width")
    parser.add_argument("-x", type=int, default=12, help="number of existential variables")
    parser.add_argument("-u", type=int, default=6, help="number of universal variables")
    parser.add_argument("-d", type=int, default=None, help="size of each dependency set")
    parser.add_argument("-s", type=float, default=0,
                        help="sample other parameters from uniform distributions on [p*(1+s), p*(1-s)]")
    args = parser.parse_args()
    main(args)
