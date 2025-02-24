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

from enum import Enum
import multiprocessing
import logging
import random
import z3
# from z3.z3util import get_vars

from arlib.quant.ufbv.reduction_types import zero_extension, right_zero_extension
from arlib.quant.ufbv.reduction_types import one_extension, right_one_extension
from arlib.quant.ufbv.reduction_types import sign_extension, right_sign_extension

process_queue = []


# glock = multiprocessing.Lock()
class Quantification(Enum):
    """Determine which variables (universally or existentially quantified)
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
        formula formula to approximate
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


def solve_with_approx_partitioned(formula_str, reduction_type, q_type, bit_places, polarity,
                                result_queue, local_max_bit_width):
    """Modified to accept formula as string"""
    # Parse formula string in worker process
    formula = z3.And(z3.parse_smt2_string(formula_str))
    
    while (bit_places < (local_max_bit_width - 2) or max_bit_width == 0):
        approximated_formula = rec_go(formula, [], reduction_type, q_type, bit_places, polarity)
        
        # Create new context and solver
        new_ctx = z3.Context()
        new_ctx_fml = approximated_formula.translate(new_ctx)
        # Fix: Create tactic and solver correctly with context
        t = z3.Tactic("ufbv", ctx=new_ctx)
        s = t.solver()
        s.add(new_ctx_fml)
        
        result = s.check()
        # Convert result to string for queue
        result_str = str(result)
        
        if q_type == Quantification.UNIVERSAL:
            if result == z3.CheckSatResult(z3.Z3_L_TRUE) or result == z3.CheckSatResult(z3.Z3_L_UNDEF):
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
            elif result == z3.CheckSatResult(z3.Z3_L_FALSE):
                result_queue.put(result_str)
                return
        else:
            if result == z3.CheckSatResult(z3.Z3_L_TRUE):
                result_queue.put(result_str)
                return
            elif result == z3.CheckSatResult(z3.Z3_L_FALSE) or result == z3.CheckSatResult(z3.Z3_L_UNDEF):
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)

    solve_without_approx(formula_str, result_queue, True)

def solve_without_approx(formula_str, result_queue, randomness=False):
    """Modified to accept formula as string"""
    formula = z3.And(z3.parse_smt2_string(formula_str))
    s = z3.Tactic("ufbv").solver()
    if randomness:
        s.set("smt.random_seed", random.randint(0, 10))
    s.add(formula)
    result_queue.put(str(s.check()))

def split_list(lst, n):
    """Split a list into n approximately equal chunks.
    
    Args:
        lst: List to split
        n: Number of chunks
    Returns:
        List of n sublists
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def solve_qbv_parallel(formula):
    # Convert formula to string
    formula_str = formula.sexpr()
    
    reduction_type = ReductionType.ZERO_EXTENSION
    timeout = 60
    workers = 4

    if workers == 1:
        return solve_sequential(formula)

    m_max_bit_width = extract_max_bits_for_formula(formula)
    partitioned_bits_lists = list(range(1, m_max_bit_width + 1))
    over_parts = split_list(partitioned_bits_lists, int(workers / 2))
    under_parts = split_list(partitioned_bits_lists, int(workers / 2))

    with multiprocessing.Manager() as manager:
        result_queue = multiprocessing.Queue()

        for nth in range(int(workers)):
            bits_id = int(nth / 2)
            if nth % 2 == 0:
                if len(over_parts[bits_id]) > 0:
                    start_width = over_parts[bits_id][0]
                    end_width = over_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                process_queue.append(
                    multiprocessing.Process(
                        target=solve_with_approx_partitioned,
                        args=(formula_str, reduction_type, Quantification.UNIVERSAL,
                              start_width, Polarity.POSITIVE, result_queue, end_width)
                    )
                )
            else:
                if len(over_parts[bits_id]) > 0:
                    start_width = under_parts[bits_id][0]
                    end_width = under_parts[bits_id][-1]
                else:
                    start_width = 1
                    end_width = m_max_bit_width
                process_queue.append(
                    multiprocessing.Process(
                        target=solve_with_approx_partitioned,
                        args=(formula_str, reduction_type, Quantification.EXISTENTIAL,
                              start_width, Polarity.POSITIVE, result_queue, end_width)
                    )
                )

        for p in process_queue:
            p.start()

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

        for p in process_queue:
            p.terminate()

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
    # TODO: if terminated by the caller, we should kill the processes in the process_queue??
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGQUIT, signal_handler)
    # signal.signal(signal.SIGHUP, signal_handler)
    demo_qbv()
