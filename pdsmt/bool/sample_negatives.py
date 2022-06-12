# coding: utf-8
import random
from pysat.formula import CNF


"""
from https://github.com/andertavares/mlbf
"""

def uniformly_negative_samples(cnf, max_samples, max_attempts=100000):
    """
    Generates negative samples of a boolean formula f uniformly at random
    :param cnf: path to the boolean formula in DIMACS CNF format
    :param max_samples: maximum number of samples to generate
    :param max_attempts: maximum number of attempts (prevents infinite loop in case of difficult negative samples)
    :return:
    """
    f = CNF(cnf)
    negatives = set()  # a set prevents the existence of duplicates
    attempts = 0
    while len(negatives) < max_samples and attempts < max_attempts:
        attempts += 1

        # generates a random assignment and tests if it satisfies f
        # the candidate follows the DIMACS format (i or -i for asserted/negated variable i)
        candidate = [x * random.choice([-1, 1]) for x in
                     range(1, f.nv + 1)]  # 1 to n+1 because literal indexes start at 1

        # avoids generating duplicate instances
        if tuple(candidate) in negatives:
            # print(f'duplicate {candidate} detected')
            continue

        # if any clause has all literals disagreeing in sign with the assignment, the candidate falsifies f
        if any([all([candidate[abs(l) - 1] * l < 0 for l in clause]) for clause in f.clauses]):
            # print(f'{candidate} evaluated to false, adding to negatives')
            negatives.add(tuple(candidate))

    # transforms the set of tuples into a list of lists
    neg_list = list([list(x) for x in negatives])

    if attempts == max_attempts:
        print(f'WARNING: maximum #attempts ({max_attempts}) to generate ~f samples reached.')

    return neg_list


def phase_transition_samples(positives, max_samples):
    """
    This method is not correct, as it does not verify if the tentative
    unsat actually falsifies the formula, it just checks if it is not in
    the positives
    :param positives:
    :param max_samples:
    :return:
    """

    unsat_list = []

    # transforming each sat instance into tuple eases the generation of negative instances
    sat_set = set([tuple(s) for s in positives])  # set is much quicker to verify an existing instance

    for assignment in positives:
        for i, literal in enumerate(assignment):
            tentative_unsat = list(assignment)
            tentative_unsat[i] = -tentative_unsat[i]  # negating one literal
            if tuple(tentative_unsat) not in sat_set:
                unsat_list.append(tentative_unsat)
                break  # goes on to next assignment
            # print(f'negated {i}-th')

    return unsat_list