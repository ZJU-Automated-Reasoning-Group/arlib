#!/usr/bin/python3
"""
Deciding quantified bit-vector (BV) formulas with abstraction

- Run under-, over- approximations in parallel
- Share information (??)

NOTE:
- s = z3.Tactic("ufbv").solver()
   If using s = z3.Solver(), z3 may use a slow tactic to create the solver!!

Some tactis for creating solvers
   - ufbv
   - bv
   - simplify & qe-light & qe2?
   - simplify & qe-light & qe?
   - simplify & smt?

TODO: Do not use Z3 to solve the concrete instances
  Dup the instances as SMT-lib2 file, and use another binary files of the solvers to solve
"""

import argparse
from enum import Enum
import multiprocessing
import logging
import random
import z3
# from z3.z3util import get_vars

from reduction_types import zero_extension, right_zero_extension
from reduction_types import one_extension, right_one_extension
from reduction_types import sign_extension, right_sign_extension

process_queue = []


# glock = multiprocessing.Lock()
class Quantification(Enum):
    """Determine which variables (universaly or existentialy quantified)
    will be approxamated.
    """
    UNIVERSAL = 0  # over-approximation
    EXISTENTIAL = 1  # under-approximation


class Polarity(Enum):
    POSITIVE = 0
    NEGATIVE = 1


class ReductionType(Enum):
    ZERO_EXTENSION = 3
    ONE_EXTENSION = 1
    SIGN_EXTENSION = 2
    RIGHT_ZERO_EXTENSION = -3
    RIGHT_ONE_EXTENSION = -1
    RIGHT_SIGN_EXTENSION = -2


max_bit_width = 0


def approximate(formula, reduction_type, bit_places):
    """Approximate given formula.

    Arguments:
        formula     formula to approximate
        reduction_type approximation type (0, 1, 2)
        bit_places  new bit width
    """
    # Do not expand smaller formulae.
    if formula.size() > bit_places:
        # Zero-extension
        if reduction_type == ReductionType.ZERO_EXTENSION:
            return zero_extension(formula, bit_places)

        # One-extension
        elif reduction_type == ReductionType.ONE_EXTENSION:
            return one_extension(formula, bit_places)

        # Sign-extension
        elif reduction_type == ReductionType.SIGN_EXTENSION:
            return sign_extension(formula, bit_places)

        # Right-zero-extension
        elif reduction_type == ReductionType.RIGHT_ZERO_EXTENSION:
            return right_zero_extension(formula, bit_places)

        # Right-one-extension
        elif reduction_type == ReductionType.RIGHT_ONE_EXTENSION:
            return right_one_extension(formula, bit_places)

        # Right-sign-extension
        elif reduction_type == ReductionType.RIGHT_SIGN_EXTENSION:
            return right_sign_extension(formula, bit_places)

        # Unknown type of approximation
        else:
            raise ValueError("Select the approximation type.")
    else:
        return formula


def recreate_vars(new_vars, formula):
    """Add quantified variables from formula to the new_vars list.
    """
    for i in range(formula.num_vars()):
        name = formula.var_name(i)

        # Type BV
        if z3.is_bv_sort(formula.var_sort(i)):
            size = formula.var_sort(i).size()
            new_vars.append(z3.BitVec(name, size))

        # Type Bool
        elif formula.var_sort(i).is_bool():
            new_vars.append(z3.Bool(name))

        else:
            raise ValueError("Unknown type of the variable:",
                             formula.var_sort(i))


def get_q_type(formula, polarity):
    """Return current quantification type.
    """
    if ((formula.is_forall() and (polarity == Polarity.POSITIVE)) or
            ((not formula.is_forall()) and (polarity == Polarity.NEGATIVE))):
        return Quantification.UNIVERSAL
    else:
        return Quantification.EXISTENTIAL


def update_vars(formula, var_list, polarity):
    """Recreate the list of quantified variables in formula and update var_list.
    """
    new_vars = []
    quantification = get_q_type(formula, polarity)

    # Add quantified variables to the var_list
    for i in range(formula.num_vars()):
        var_list.append((formula.var_name(i), quantification))

    # Recreate list of variables bounded by this quantifier
    recreate_vars(new_vars, formula)

    # Sequentialy process following quantifiers
    while ((type(formula.body()) == z3.QuantifierRef) and
           ((formula.is_forall() and formula.body().is_forall()) or
            (not formula.is_forall() and not formula.body().is_forall()))):
        for i in range(formula.body().num_vars()):
            var_list.append((formula.body().var_name(i), quantification))
        recreate_vars(new_vars, formula.body())
        formula = formula.body()

    return new_vars, formula


def qform_process(formula, var_list, reduction_type,
                  q_type, bit_places, polarity):
    """Create new quantified formula with modified body.
    """
    # Recreate the list of quantified variables and update current formula
    new_vars, formula = update_vars(formula, var_list, polarity)

    # Recursively process the body of the formula and create the new body
    new_body = rec_go(formula.body(),
                      list(var_list),
                      reduction_type,
                      q_type,
                      bit_places,
                      polarity)

    # Create new quantified formula with modified body
    if formula.is_forall():
        formula = z3.ForAll(new_vars, new_body)
    else:
        formula = z3.Exists(new_vars, new_body)

    return formula


def cform_process(formula, var_list, reduction_type, q_type, bit_places,
                  polarity):
    """Process individual parts of a compound formula and recreate the formula.
    """
    new_children = []
    var_list_copy = list(var_list)

    # Negation: Switch the polarity
    if formula.decl().name() == "not":
        polarity = Polarity(not polarity.value)

    # Implication: Switch polarity
    elif formula.decl().name() == "=>":
        # Switch polarity just for the left part of implication
        polarity2 = Polarity(not polarity.value)

        new_children.append(rec_go(formula.children()[0],
                                   var_list_copy,
                                   reduction_type,
                                   q_type,
                                   bit_places,
                                   polarity2))
        new_children.append(rec_go(formula.children()[1],
                                   var_list_copy,
                                   reduction_type,
                                   q_type,
                                   bit_places,
                                   polarity))
        return z3.Implies(*new_children)

    # Recursively process children of the formula
    for i in range(len(formula.children())):
        new_children.append(rec_go(formula.children()[i],
                                   var_list_copy,
                                   reduction_type,
                                   q_type,
                                   bit_places,
                                   polarity))

    # Recreate trouble making operands with arity greater then 2
    if formula.decl().name() == "and":
        formula = z3.And(*new_children)

    elif formula.decl().name() == "or":
        formula = z3.Or(*new_children)

    elif formula.decl().name() == "bvadd":
        formula = new_children[0]
        for ch in new_children[1::]:
            formula = formula + ch

    # Recreate problem-free operands
    else:
        formula = formula.decl().__call__(*new_children)

    return formula


def rec_go(formula, var_list, reduction_type, q_type, bit_places, polarity):
    """Recursively go through the formula and apply approximations.
    """
    # Constant
    if z3.is_const(formula):
        pass

    # Variable
    elif z3.is_var(formula):
        order = - z3.get_var_index(formula) - 1

        # Process if it is bit-vector variable
        if type(formula) == z3.BitVecRef:
            # Update max bit-vector width
            global max_bit_width
            if max_bit_width < formula.size():
                max_bit_width = formula.size()
            # print(max_bit_width)
            # Approximate if var is quantified in the right way
            if var_list[order][1] == q_type:
                formula = approximate(formula, reduction_type, bit_places)

    # Quantified formula
    elif type(formula) == z3.QuantifierRef:
        formula = qform_process(formula,
                                list(var_list),
                                reduction_type,
                                q_type,
                                bit_places,
                                polarity)

    # Complex formula
    else:
        formula = cform_process(formula,
                                list(var_list),
                                reduction_type,
                                q_type,
                                bit_places,
                                polarity)

    # print("max bit width: ", max_bit_width)

    return formula


def get_max_bit_width():
    return max_bit_width


def next_approx(reduction_type, bit_places):
    """Change reduction type and increase the bit width.
    """
    # Switch left/right reduction
    reduction_type = ReductionType(- reduction_type.value)

    # Resize bit width
    if reduction_type.value < 0:
        if bit_places == 1:
            bit_places += 1
        else:
            bit_places += 2

    return reduction_type, bit_places


def extract_max_bits_for_formula(fml):
    reduction_type = ReductionType.ONE_EXTENSION
    # q_type = Quantification.UNIVERSAL
    q_type = Quantification.EXISTENTIAL
    bit_places = 1
    polarity = Polarity.POSITIVE
    rec_go(fml,
           [],
           reduction_type,
           q_type,
           bit_places,
           polarity)
    return get_max_bit_width()


def solve_with_approx_partitioned(formula, reduction_type, q_type, bit_places, polarity,
                                  result_queue, local_max_bit_width):
    """Recursively go through `formula` and approximate it. Check
    satisfiability of the approximated formula. Put the result to
    the `result_queue`.
    """
    # print(local_max_bit_width)
    # Approximate until maximal bit width is reached
    # TODO: what if local_max_bit_width - 2 < 0?
    while (bit_places < (local_max_bit_width - 2) or
           max_bit_width == 0):

        # Approximate the formula
        approximated_formula = rec_go(formula,
                                      [],
                                      reduction_type,
                                      q_type,
                                      bit_places,
                                      polarity)

        # logging.debug("approximation generation success!")
        # Solve the approximated formula
        # s = z3.Solver()
        s = z3.Tactic("ufbv").solver()
        s.add(approximated_formula)
        result = s.check()
        if q_type == Quantification.UNIVERSAL:
            logging.debug("over-appro solving finished")
        else:
            logging.debug("under-appro solving finished")

        # print("approximation solving result: ", result)

        # Continue with approximation or return the result
        if q_type == Quantification.UNIVERSAL:
            # Over-approximation of the formula is SAT or unknown
            # TODO: when over-approx is SAT, we may try the model to see if it
            #   really satisfies the orignal formula (not easy to implement as \
            #   the vars have been changed
            # Approximation continues
            if (result == z3.CheckSatResult(z3.Z3_L_TRUE) or
                    result == z3.CheckSatResult(z3.Z3_L_UNDEF)):
                # Update reduction type and increase bit width
                (reduction_type, bit_places) = next_approx(reduction_type,
                                                           bit_places)

            # Over-approximation of the formula is UNSAT
            # Original formula is UNSAT
            elif result == z3.CheckSatResult(z3.Z3_L_FALSE):
                logging.debug('over-appro success')
                result_queue.put(result)
                return
        else:
            # Under-approximation of the formula is SAT
            # Original formula is SAT
            if result == z3.CheckSatResult(z3.Z3_L_TRUE):
                logging.debug('under-appro success')
                result_queue.put(result)
                return

            # Under-approximation of the formula is UNSAT or unknown
            # Approximation continues
            elif (result == z3.CheckSatResult(z3.Z3_L_FALSE) or
                  result == z3.CheckSatResult(z3.Z3_L_UNDEF)):

                # Update reduction type and increase bit width
                (reduction_type, bit_places) = next_approx(reduction_type,
                                                           bit_places)

    solve_without_approx(formula, result_queue,
                         True)  # why do we need this? because the appro cannot cover all sub-spaces?


def solve_without_approx(formula, result_queue, randomness=False):
    """Solve the given formula without any preprocessing.
    """
    # TODO: some tactic may return `unknown'
    logging.debug('solve without approximation')
    s = z3.Tactic("ufbv").solver()
    if randomness: s.set("smt.random_seed", random.randint(0, 10))
    s.add(formula)
    result_queue.put(s.check())


def solve_sequential(formula):
    # z3.set_param("verbose", 15)
    # s = z3.Solver()
    s = z3.Tactic("ufbv").solver()
    # TODO: try more tactics, e.g., 'bv', 'smt', 'qe2'...
    s.add(formula)
    print(s.check())


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def signal_handler(sig, frame):
    global process_queue
    try:
        # glock.acquire()
        for p in process_queue:
            if p: p.terminate()
        # print("processes cleaned!!")
        logging.debug("processes cleaned!")
    except Exception as e:
        # glock.release()
        pass
    # finally:
    #    glock.release() 
    # print("We finish here, have a good day!")


def main_multiple():
    global process_queue

    parser = argparse.ArgumentParser(description="Solve given formula.")
    parser.add_argument("formula", metavar="formula_file", type=str,
                        help="path to .smt2 formula file")
    parser.add_argument("-r", "--reduction", default="0", type=str,
                        help="determine the reduction type (0/1/s)")

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    parser.add_argument('--workers', dest='workers', default=3, type=int, help="num threads")

    parser.add_argument('--timeout', dest='timeout', default=60, type=int, help="timeout")

    parser.add_argument('--strategy', dest='strategy', default='default', type=str,
                        help="strategy for generation: default, another")

    args = parser.parse_args()

    if args.verbose: logging.basicConfig(level=logging.DEBUG)

    # Determine the type of reduction

    if args.reduction == "1":
        reduction_type = ReductionType.ONE_EXTENSION
    elif args.reduction == "s":
        reduction_type = ReductionType.SIGN_EXTENSION
    else:
        reduction_type = ReductionType.ZERO_EXTENSION

    # File with .smt2 formula
    formula_file = args.formula

    # Parse SMT2 formula to Z3 format
    formula = z3.And(z3.parse_smt2_file(formula_file))

    # TODO: Not correct, should consider quant?

    if args.strategy == "seq" or args.workers == 1:
        solve_sequential(formula)
        exit(0)

    # Get max bit
    # TODO: is the computed width correct???
    m_max_bit_width = extract_max_bits_for_formula(formula)
    # m_max_bit_width = 24
    partitioned_bits_lists = []
    logging.debug("max width: {}".format(m_max_bit_width))

    for i in range(1, m_max_bit_width + 1):
        partitioned_bits_lists.append(i)

    # logging.debug("max bit: ", m_max_bit_width)
    over_parts = split_list(partitioned_bits_lists, int(args.workers / 2))
    under_parts = split_list(partitioned_bits_lists, int(args.workers / 2))

    # logging.debug(over_parts)
    # logging.debug(under_parts)

    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()

        # process_queue = []

        for nth in range(int(args.workers)):
            bits_id = int(nth / 2)
            if nth % 2 == 0:
                if len(over_parts[bits_id]) > 0:
                    start_width = over_parts[bits_id][0]
                    end_width = over_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                # Over-approximation workers
                process_queue.append(multiprocessing.Process(target=solve_with_approx_partitioned,
                                                             args=(formula,
                                                                   reduction_type,
                                                                   Quantification.UNIVERSAL,
                                                                   start_width,
                                                                   # 1,
                                                                   Polarity.POSITIVE,
                                                                   result_queue,
                                                                   end_width
                                                                   # m_max_bit_width
                                                                   )))
            else:
                if len(over_parts[bits_id]) > 0:
                    start_width = under_parts[bits_id][0]
                    end_width = under_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                # Under-approximation workers
                process_queue.append(multiprocessing.Process(target=solve_with_approx_partitioned,
                                                             args=(formula,
                                                                   reduction_type,
                                                                   Quantification.EXISTENTIAL,
                                                                   start_width,
                                                                   # 1,
                                                                   Polarity.POSITIVE,
                                                                   result_queue,
                                                                   end_width,
                                                                   # m_max_bit_width
                                                                   )))

        # Start all
        for p in process_queue:
            p.start()

        # Get result
        try:
            # Wait at most 60 seconds for a return
            result = result_queue.get(timeout=int(args.timeout))
        except multiprocessing.queues.Empty:
            # If queue is empty, set result to undef
            result = z3.CheckSatResult(z3.Z3_L_UNDEF)

        # Terminate all
        for p in process_queue:
            p.terminate()

    print(result)
    return result


if __name__ == "__main__":
    # TODO: if terminated by the caller, we should kill the processes in the process_queue??
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGQUIT, signal_handler)
    # signal.signal(signal.SIGHUP, signal_handler)

    main_multiple()
