#!/usr/bin/python3
# coding: utf-8
"""
Deciding quantified bit-vector (BV) formulas with abstraction

- Run under-, over- approximations in parallel
- Share information (??)

NOTE:
- s = z3.Tactic("ufbv").solver()
   If using s = z3.Solver(), z3 may use a slow tactic to create the solver!!

Some tactics for creating solvers
   - ufbv
   - bv
   - simplify & qe-light & qe2?
   - simplify & qe-light & qe?
   - simplify & smt?
"""

# import argparse
from enum import Enum
import multiprocessing
# import sys
import logging

import z3
from z3 import BitVecVal, Concat, Extract


def zero_extension(formula, bit_places):
    """Set the rest of bits on the left to 0.
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))

    return formula


def one_extension(formula, bit_places):
    """Set the rest of bits on the left to 1.
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))

    return formula


def sign_extension(formula, bit_places):
    """Set the rest of bits on the left to the value of the sign bit.
    """
    sign_bit = Extract(bit_places - 1, bit_places - 1, formula)

    complement = sign_bit
    for _ in range(formula.size() - bit_places - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))

    return formula


def right_zero_extension(formula, bit_places):
    """Set the rest of bits on the right to 0.
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)

    return formula


def right_one_extension(formula, bit_places):
    """Set the rest of bits on the right to 1.
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)

    return formula


def right_sign_extension(formula, bit_places):
    """Set the rest of bits on the right to the value of the sign bit.
    """
    sign_bit_position = formula.size() - bit_places
    sign_bit = Extract(sign_bit_position, sign_bit_position, formula)

    complement = sign_bit
    for _ in range(sign_bit_position - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(Extract(formula.size() - 1,
                             sign_bit_position,
                             formula),
                     complement)

    return formula


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

    return formula


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


def solve_with_approx(formula_str, reduction_type, q_type, bit_places, polarity, result_queue):
    """Modified to accept formula as string instead of Z3 formula object"""
    # Parse formula string in the worker process
    formula = z3.And(z3.parse_smt2_string(formula_str))

    while (bit_places < (max_bit_width - 2) or max_bit_width == 0):
        approximated_formula = rec_go(formula, [], reduction_type, q_type, bit_places, polarity)

        logging.debug("approximation generation success!")
        s = z3.Tactic("ufbv").solver()
        s.add(approximated_formula)
        result = s.check()

        if q_type == Quantification.UNIVERSAL:
            if (result == z3.CheckSatResult(z3.Z3_L_TRUE) or
                    result == z3.CheckSatResult(z3.Z3_L_UNDEF)):
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
            elif result == z3.CheckSatResult(z3.Z3_L_FALSE):
                result_queue.put(str(result))  # Convert result to string
                logging.debug('over-appro success')
                return
        else:
            if result == z3.CheckSatResult(z3.Z3_L_TRUE):
                result_queue.put(str(result))  # Convert result to string
                logging.debug('under-appro success')
                return
            elif (result == z3.CheckSatResult(z3.Z3_L_FALSE) or
                  result == z3.CheckSatResult(z3.Z3_L_UNDEF)):
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)

    solve_without_approx(formula_str, result_queue)


def solve_without_approx(formula_str, result_queue):
    """Modified to accept formula as string"""
    logging.debug('solve without approximation')
    formula = z3.And(z3.parse_smt2_string(formula_str))
    s = z3.Tactic("ufbv").solver()
    s.add(formula)
    result_queue.put(str(s.check()))  # Convert result to string


def solve_qbv_parallel(formula):
    # Convert formula to string for passing to workers
    formula_str = formula.sexpr()

    reduction_type = ReductionType.ZERO_EXTENSION
    timeout = 1200  # 20 minutes timeout

    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()

        # Create processes with formula string
        p0 = multiprocessing.Process(target=solve_without_approx,
                                     args=(formula_str, result_queue))

        p1 = multiprocessing.Process(target=solve_with_approx,
                                     args=(formula_str, reduction_type,
                                           Quantification.UNIVERSAL, 1,
                                           Polarity.POSITIVE, result_queue))

        p2 = multiprocessing.Process(target=solve_with_approx,
                                     args=(formula_str, reduction_type,
                                           Quantification.EXISTENTIAL, 1,
                                           Polarity.POSITIVE, result_queue))

        # Start processes
        p0.start()
        p1.start()
        p2.start()

        try:
            result_str = result_queue.get(timeout=timeout)
            # Convert string result back to Z3 result
            if result_str == "sat":
                result = z3.CheckSatResult(z3.Z3_L_TRUE)
            elif result_str == "unsat":
                result = z3.CheckSatResult(z3.Z3_L_FALSE)
            else:
                result = z3.CheckSatResult(z3.Z3_L_UNDEF)
        except multiprocessing.queues.Empty:
            result = z3.CheckSatResult(z3.Z3_L_UNDEF)

        # Terminate processes
        p0.terminate()
        p1.terminate()
        p2.terminate()

    print(result)
    return result


def solve_qbv_file_parallel(formula_file: str):
    # Parse SMT2 formula to Z3 format
    formula = z3.And(z3.parse_smt2_file(formula_file))
    return solve_qbv_parallel(formula)


def solve_qbv_str_parallel(fml_str: str):
    # Parse SMT2 formula to Z3 format
    formula = z3.And(z3.parse_smt2_string(fml_str))
    return solve_qbv_parallel(formula)


def demo_qbv():
    fml_str = ''' \n
(assert \n
 (exists ((s (_ BitVec 5)) )(forall ((t (_ BitVec 5)) )(not (= (bvnand s t) (bvor s (bvneg t))))) \n
 ) \n
 ) \n
(check-sat)
'''
    solve_qbv_str_parallel(fml_str)


if __name__ == "__main__":
    demo_qbv()
