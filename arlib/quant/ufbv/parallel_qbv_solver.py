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
import tempfile
import subprocess
import os
import signal
import atexit
# from z3.z3util import get_vars

from arlib.quant.ufbv.reduction_types import zero_extension, right_zero_extension
from arlib.quant.ufbv.reduction_types import one_extension, right_one_extension
from arlib.quant.ufbv.reduction_types import sign_extension, right_sign_extension

from arlib.global_params import global_config

# Path to Z3 executable
Z3_PATH = global_config.get_solver_path("z3")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

process_queue = []

# Setup handlers to clean up processes on exit
def cleanup_processes():
    for p in process_queue:
        if p.is_alive():
            p.terminate()

atexit.register(cleanup_processes)

def signal_handler(signum, frame):
    """Handle signals to clean up processes"""
    cleanup_processes()
    exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGQUIT'):
    signal.signal(signal.SIGQUIT, signal_handler)
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, signal_handler)

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
    try:
        formula = z3.And(z3.parse_smt2_string(formula_str))
    except Exception as e:
        logger.error(f"Error parsing formula: {e}")
        result_queue.put("unknown")
        return

    while (bit_places < (local_max_bit_width - 2) or max_bit_width == 0):
        try:
            approximated_formula = rec_go(formula, [], reduction_type, q_type, bit_places, polarity)
            
            temp_path = None
            try:
                # Use external Z3 process instead of internal Z3 API
                with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
                    temp_path = temp_file.name
                    formula_text = approximated_formula.sexpr()
                    # Add option to disable traceback and debug messages
                    temp_file.write("(set-option :print-success false)\n")
                    temp_file.write("(set-option :produce-unsat-cores false)\n")
                    temp_file.write("(set-option :produce-proofs false)\n")
                    temp_file.write("(set-option :trace false)\n")
                    temp_file.write("(set-option :verbose 0)\n")
                    temp_file.write(formula_text)
                    if "(check-sat)" not in formula_text:
                        temp_file.write("\n(check-sat)")
                
                logger.debug(f"Running Z3 with bit width {bit_places}, reduction {reduction_type}, q_type {q_type}")
                result = subprocess.run(
                    [Z3_PATH, "-smt2", "-q", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse result - look for the last line containing sat/unsat/unknown
                output_lines = result.stdout.strip().split('\n')
                result_str = "unknown"
                for line in output_lines:
                    line = line.strip()
                    if line in ["sat", "unsat", "unknown"]:
                        result_str = line
                
                if q_type == Quantification.UNIVERSAL:
                    if result_str == "sat" or result_str == "unknown":
                        (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
                    elif result_str == "unsat":
                        result_queue.put(result_str)
                        return
                else:
                    if result_str == "sat":
                        result_queue.put(result_str)
                        return
                    elif result_str == "unsat" or result_str == "unknown":
                        (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
            except subprocess.TimeoutExpired:
                logger.warning(f"Z3 timeout with bit width {bit_places}")
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
            except Exception as e:
                logger.error(f"Error running Z3: {e}")
                (reduction_type, bit_places) = next_approx(reduction_type, bit_places)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error in approximation: {e}")
            (reduction_type, bit_places) = next_approx(reduction_type, bit_places)

    # If we got here, we've run out of approximations to try
    # Fall back to a direct solve with the original formula
    solve_without_approx(formula_str, result_queue, True)


def solve_without_approx(formula_str, result_queue, randomness=False):
    """Modified to accept formula as string and use external Z3 process"""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
            # Make sure formula_str contains SMT-LIB content
            # Add option to disable traceback and debug messages
            temp_file.write("(set-option :print-success false)\n")
            temp_file.write("(set-option :produce-unsat-cores false)\n")
            temp_file.write("(set-option :produce-proofs false)\n")
            temp_file.write("(set-option :trace false)\n")
            # Set quiet mode
            temp_file.write("(set-option :global-decls false)\n")
            temp_file.write("(set-option :verbose 0)\n")
            temp_file.write(formula_str)
            if "(check-sat)" not in formula_str:
                temp_file.write("\n(check-sat)")
            
            if randomness:
                # Set random seed in SMT2 format
                rand_seed = random.randint(0, 10)
                temp_file.write(f"\n(set-option :smt.random_seed {rand_seed})")
        
        logger.debug("Running direct Z3 solve without approximation")
        result = subprocess.run(
            [Z3_PATH, "-smt2", "-q", temp_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse result - look for the last line containing sat/unsat/unknown
        output_lines = result.stdout.strip().split('\n')
        result_str = "unknown"
        for line in output_lines:
            line = line.strip()
            if line in ["sat", "unsat", "unknown"]:
                result_str = line
                
        logger.debug(f"Z3 result: {result_str}")
        
        if result_str == "sat":
            result_queue.put("sat")
        elif result_str == "unsat":
            result_queue.put("unsat")
        else:
            result_queue.put("unknown")
    except subprocess.TimeoutExpired:
        logger.warning("Z3 direct solve timeout")
        result_queue.put("unknown")
    except Exception as e:
        logger.error(f"Error in solve_without_approx: {e}")
        result_queue.put("unknown")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


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


def solve_sequential(formula):
    """Solve the formula without parallelization using external Z3 process.
    
    This is used as a fallback when workers=1.
    """
    formula_str = formula.sexpr()
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
            # Add option to disable traceback and debug messages
            temp_file.write("(set-option :print-success false)\n")
            temp_file.write("(set-option :produce-unsat-cores false)\n")
            temp_file.write("(set-option :produce-proofs false)\n")
            temp_file.write("(set-option :trace false)\n")
            # Set quiet mode
            temp_file.write("(set-option :global-decls false)\n")
            temp_file.write("(set-option :verbose 0)\n")
            temp_file.write(formula_str)
            if "(check-sat)" not in formula_str:
                temp_file.write("\n(check-sat)")
    
        logger.debug("Running sequential solve without parallelization")
        result = subprocess.run(
            [Z3_PATH, "-smt2", "-q", temp_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Parse result - look for the last line containing sat/unsat/unknown
        output_lines = result.stdout.strip().split('\n')
        result_str = "unknown"
        for line in output_lines:
            line = line.strip()
            if line in ["sat", "unsat", "unknown"]:
                result_str = line
        
        logger.debug(f"Z3 sequential result: {result_str}")
        
        if result_str == "sat":
            return z3.CheckSatResult(z3.Z3_L_TRUE)
        elif result_str == "unsat":
            return z3.CheckSatResult(z3.Z3_L_FALSE)
        else:
            return z3.CheckSatResult(z3.Z3_L_UNDEF)
    except subprocess.TimeoutExpired:
        logger.warning("Z3 sequential solve timeout")
        return z3.CheckSatResult(z3.Z3_L_UNDEF)
    except Exception as e:
        logger.error(f"Error in solve_sequential: {e}")
        return z3.CheckSatResult(z3.Z3_L_UNDEF)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


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
    """Parse and solve formula from an SMT2 file using parallel approximation
    
    Args:
        formula_file: Path to the SMT2 file
    
    Returns:
        Z3 CheckSatResult (sat, unsat, or unknown)
    """
    try:
        # Read SMT2 file content directly instead of parsing with Z3
        with open(formula_file, 'r') as f:
            formula_str = f.read()
        
        # Check if it's a valid SMT2 file
        if not formula_str or not ("(assert" in formula_str):
            logger.error(f"Invalid SMT2 file: {formula_file}")
            return z3.CheckSatResult(z3.Z3_L_UNDEF)
        
        # Parse to Z3 format for max bit width extraction
        formula = z3.And(z3.parse_smt2_file(formula_file))
        
        # Pass the raw SMT2 string to the solver
        return solve_qbv_parallel(formula)
    except Exception as e:
        logger.error(f"Error in solve_qbv_file_parallel: {e}")
        return z3.CheckSatResult(z3.Z3_L_UNDEF)


def solve_qbv_str_parallel(fml_str: str):
    """Parse and solve formula from an SMT2 string using parallel approximation
    
    Args:
        fml_str: SMT2 format string
    
    Returns:
        Z3 CheckSatResult (sat, unsat, or unknown)
    """
    try:
        # Verify it's a valid SMT2 string
        if not fml_str or not ("(assert" in fml_str):
            # Add assert if missing
            if not "(assert" in fml_str:
                fml_str = f"(assert {fml_str})"
        
        # Parse to Z3 format for max bit width extraction
        formula = z3.And(z3.parse_smt2_string(fml_str))
        
        # Pass the formula to the solver
        return solve_qbv_parallel(formula)
    except Exception as e:
        logger.error(f"Error in solve_qbv_str_parallel: {e}")
        return z3.CheckSatResult(z3.Z3_L_UNDEF)


def demo_qbv():
    """Demonstrate the parallel QBV solver with some examples"""
    try:
        print("=" * 60)
        print("DEMO: Parallel QBV Solver with External Z3")
        print("=" * 60)
        
        # Example 1: BV NAND/OR formula
        print("\nExample 1: BV NAND/OR formula")
        fml_str = '''(assert 
 (exists ((s (_ BitVec 5)) )(forall ((t (_ BitVec 5)) )(not (= (bvnand s t) (bvor s (bvneg t))))) 
 ) 
 )
(check-sat)'''
        print(f"Formula: {fml_str}")
        result = solve_qbv_str_parallel(fml_str)
        print(f"Result: {result}")
        
        # Example 2: Simple formula
        print("\nExample 2: Simple BV formula")
        simple_fml_str = '''(assert 
 (exists ((x (_ BitVec 4))) (forall ((y (_ BitVec 4))) (= (bvadd x y) (bvadd y x)))
 )
)
(check-sat)'''
        print(f"Formula: {simple_fml_str}")
        result = solve_qbv_str_parallel(simple_fml_str)
        print(f"Result: {result}")
        
        # Example 3: Create a temporary file
        print("\nExample 3: Using a formula file")
        with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
            file_content = '''(assert 
 (exists ((x (_ BitVec 8))) 
   (forall ((y (_ BitVec 8))) 
     (=> (bvult y x) (exists ((z (_ BitVec 8))) (= (bvadd y z) x)))
   )
 )
)
(check-sat)'''
            temp_file.write(file_content)
        
        try:
            print(f"Formula file: {temp_path}")
            print(f"Formula content:\n{file_content}")
            result = solve_qbv_file_parallel(temp_path)
            print(f"Result: {result}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        print("\nDemo complete")
    except Exception as e:
        logger.error(f"Error in demo: {e}")


if __name__ == "__main__":
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Parallel QBV Solver with external Z3')
        
        # Add arguments
        parser.add_argument('--file', type=str, help='SMT2 file to solve')
        parser.add_argument('--formula', type=str, help='SMT2 formula string to solve')
        parser.add_argument('--demo', action='store_true', help='Run demo examples')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        
        args = parser.parse_args()
        
        # Set logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Determine what to run
        if args.file:
            print(f"Solving file: {args.file}")
            result = solve_qbv_file_parallel(args.file)
            print(f"Result: {result}")
        elif args.formula:
            print(f"Solving formula: {args.formula}")
            result = solve_qbv_str_parallel(args.formula)
            print(f"Result: {result}")
        elif args.demo:
            demo_qbv()
        else:
            # Default to demo
            demo_qbv()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Make sure we clean up any remaining processes
        cleanup_processes()
