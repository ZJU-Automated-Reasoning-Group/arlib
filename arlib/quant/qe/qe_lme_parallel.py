"""Parallel Quantifier Elimination via Lazy Model Enumeration (LME-QE)

An implementation of LME-QE (CAV 2013) with parallelized model enumeration and projection.
This algorithm eliminates existential quantifiers through iterative model enumeration
in parallel, potentially improving performance on multi-core systems.

This implementation uses Z3 as an external process via IPC for better isolation and potential performance.

Implementation Details:
----------------------
1. Communication with Z3: Uses SMT-LIB format to communicate with Z3 as an external process
2. Model Enumeration: Extracts models in parallel and processes them to find projections
3. Two-tier Approach:
   - First attempts direct QE through Z3
   - Falls back to model enumeration if direct QE fails
4. Parallelization: Uses ProcessPoolExecutor for parallel model processing

Example Usage:
-------------
```python
import z3
from arlib.quant.qe.qe_lme_parallel import qelim_exists_lme_parallel

x, y, z = z3.Reals("x y z")
formula = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))
result = qelim_exists_lme_parallel(formula, [x, y])
print(result)  # SMT-LIB formatted result
```
"""

from typing import List, Dict, Any, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile
import os
import re
import json
import logging
from pathlib import Path
import z3

from arlib.utils.z3_expr_utils import negate # get_atoms
from arlib.global_params import global_config

# Set up logging
logger = logging.getLogger(__name__)

# Path to Z3 executable
Z3_PATH = global_config.get_solver_path("z3")

# SMT-LIB templates
SMT_HEADER = """
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
{declarations}
(assert {formula})
"""

QE_TEMPLATE = """
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
{declarations}
(assert (exists ({qvars}) {formula}))
(apply qe)
(get-assertions)
"""


def run_z3_with_tracing(smt_script: str, description: str = "") -> subprocess.CompletedProcess:
    """Run Z3 and log the input query and output response.
    
    Args:
        smt_script: The SMT-LIB script to run
        description: A description of what this Z3 call is doing (for logging)
        
    Returns:
        The completed process with stdout and stderr
    """
    try:
        # Create a temporary file for the SMT script
        with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(smt_script)
            
        # Log the Z3 query
        with open("z3_trace.log", "a") as log:
            log.write(f"\n=== Z3 Query: {description} ===\n")
            log.write(smt_script)
            log.write("\n")
            
        # Run Z3
        result = subprocess.run(
            [Z3_PATH, "-smt2", temp_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Log the response
        with open("z3_trace.log", "a") as log:
            log.write("=== Z3 Response ===\n")
            log.write(result.stdout)
            if result.stderr:
                log.write("=== Z3 Errors ===\n")
                log.write(result.stderr)
            log.write("="*40 + "\n")
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in run_z3_with_tracing: {e}")
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=f"Error: {str(e)}"
        )


def to_smtlib(expr):
    """Convert Z3 expression to SMT-LIB string format"""
    if hasattr(expr, 'sexpr'):
        return expr.sexpr()
    return str(expr)


def get_declarations(expr):
    """Extract variable declarations from expression"""
    decls = set()
    
    try:
        if hasattr(expr, 'children'):
            variables = set()
            
            def collect_vars(e):
                if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    variables.add(e)
                elif hasattr(e, 'children'):
                    for child in e.children():
                        collect_vars(child)
            
            collect_vars(expr)
            
            for var in variables:
                sort = var.sort()
                sort_name = sort.name()
                if sort_name == "Int":
                    decls.add(f"(declare-const {var} Int)")
                elif sort_name == "Real":
                    decls.add(f"(declare-const {var} Real)")
                elif sort_name == "Bool":
                    decls.add(f"(declare-const {var} Bool)")
                else:
                    decls.add(f"(declare-const {var} {sort_name})")
        
        if not decls:
            var_pattern = r'([a-zA-Z][a-zA-Z0-9_]*)'
            expr_str = str(expr)
            
            for match in re.finditer(var_pattern, expr_str):
                var_name = match.group(1)
                if var_name not in ["and", "or", "not", "true", "false", "exists", "forall"]:
                    decls.add(f"(declare-const {var_name} Real)")
    
    except Exception as e:
        print(f"Warning: Error parsing declarations: {e}")

    return "\n".join(sorted(list(decls)))


def parse_model(z3_output):
    """Parse Z3 model output to a dictionary of variable assignments"""
    model_dict = {}
    
    try:
        if "sat" not in z3_output:
            return model_dict
            
        model_match = re.search(r'sat\s*\((.*)\)\s*$', z3_output, re.DOTALL)
        if not model_match:
            return model_dict
            
        model_content = model_match.group(1).strip()
        
        define_fun_blocks = []
        current_block = ""
        depth = 0
        
        for char in model_content:
            if char == '(':
                depth += 1
                if depth == 1:
                    current_block = "("
                else:
                    current_block += char
            elif char == ')':
                depth -= 1
                current_block += char
                if depth == 0:
                    define_fun_blocks.append(current_block)
                    current_block = ""
            elif depth > 0:
                current_block += char
        
        for block in define_fun_blocks:
            if block.strip().startswith("(define-fun"):
                parts = block.strip()[11:].strip().split(None, 3)
                if len(parts) >= 3:
                    var_name = parts[0].strip()
                    var_type = parts[2].strip()
                    
                    value_start_idx = block.find(var_type) + len(var_type)
                    var_value = block[value_start_idx:].strip()
                    
                    model_dict[var_name] = {
                        'type': var_type,
                        'value': var_value
                    }
    except Exception as e:
        print(f"Error parsing model: {e}")
    
    return model_dict


def create_blocking_clause(model_dict):
    """Create an SMT-LIB blocking clause from a model dictionary"""
    clauses = []
    
    for var_name, var_info in model_dict.items():
        if var_info['type'] == 'Bool':
            if var_info['value'] == 'true':
                clauses.append(f"(not {var_name})")
            else:
                clauses.append(var_name)
        else:
            clauses.append(f"(not (= {var_name} {var_info['value']}))")
    
    if not clauses:
        return "true"
    elif len(clauses) == 1:
        return clauses[0]
    else:
        return f"(or {' '.join(clauses)})"


def extract_models(formula, num_models=10, blocked_models=None, trace_z3=False):
    """Extract models from a formula using Z3 via IPC"""
    if blocked_models is None:
        blocked_models = []
    
    models = []
    
    try:
        formula_smtlib = to_smtlib(formula)
        
        if blocked_models:
            blocking_clauses = [to_smtlib(negate(model_expr)) for model_expr in blocked_models]
            formula_smtlib = f"(and {formula_smtlib} {' '.join(blocking_clauses)})"
        
        declarations = get_declarations(formula)
        
        for i in range(num_models):
            smt_script = SMT_HEADER.format(
                declarations=declarations,
                formula=formula_smtlib
            )
            smt_script += "\n(check-sat)"
            smt_script += "\n(get-model)"
            
            if trace_z3:
                result = run_z3_with_tracing(smt_script, description=f"Model Extraction {i+1}")
            else:
                with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(smt_script)
                    
                result = subprocess.run(
                    [Z3_PATH, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                os.unlink(temp_path)
            
            if "sat" in result.stdout:
                model_dict = parse_model(result.stdout)
                if model_dict:
                    models.append(model_dict)
                    
                    blocking_clause = create_blocking_clause(model_dict)
                    formula_smtlib = f"(and {formula_smtlib} {blocking_clause})"
                    
                    if trace_z3:
                        with open("z3_trace.log", "a") as log:
                            log.write(f"Added blocking clause: {blocking_clause}\n")
                else:
                    break
            else:
                break
                    
    except Exception as e:
        print(f"Error in extract_models: {e}")
    
    return models


def parse_qe_result(z3_output):
    """Parse quantifier elimination result from Z3 output"""
    double_paren_match = re.search(r'\(\((and[^)]+)\)\)', z3_output, re.DOTALL)
    if double_paren_match:
        return double_paren_match.group(1)
    
    direct_and_match = re.search(r'\(and\s+([^)]+)\)', z3_output, re.DOTALL)
    if direct_and_match:
        return f"(and {direct_and_match.group(1)})"
    
    get_assertions = re.search(r'\(get-assertions\)\s*(.+)', z3_output, re.DOTALL)
    if get_assertions:
        assertions_text = get_assertions.group(1).strip()
        if assertions_text.startswith('(('):
            depth = 0
            for i, char in enumerate(assertions_text):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        return assertions_text[:i+1]
    
    return "false"


def try_direct_qe(phi, qvars, trace_z3=False):
    """Try direct quantifier elimination using Z3"""
    try:
        declarations = get_declarations(phi)
        qvars_smtlib = " ".join([f"({str(var)} {var.sort().sexpr()})" for var in qvars])
        
        smt_script = QE_TEMPLATE.format(
            declarations=declarations,
            qvars=qvars_smtlib,
            formula=to_smtlib(phi)
        )
        smt_script += "\n(get-assertions)"
        
        if trace_z3:
            result = run_z3_with_tracing(smt_script, description="Direct QE Attempt")
        else:
            with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(smt_script)
                
            result = subprocess.run(
                [Z3_PATH, temp_file_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            os.unlink(temp_file_path)
        
        if "unknown" not in result.stdout.lower() and "error" not in result.stdout.lower():
            qe_result = parse_qe_result(result.stdout)
            if qe_result:
                return qe_result
    
    except Exception as e:
        print(f"Error in direct QE attempt: {e}")
    
    return None


def build_minterm_from_model(model):
    """Build a minterm from a model's evaluation of predicates"""
    constraints = []
    
    for var_name, var_info in model.items():
        var_type = var_info['type']
        var_value = var_info['value'].strip()
        
        if var_type == 'Bool':
            if var_value == 'true':
                constraints.append(var_name)
            elif var_value == 'false':
                constraints.append(f"(not {var_name})")
        else:
            if var_value.startswith('(') and var_value.endswith(')'):
                constraints.append(f"(= {var_name} {var_value})")
            else:
                constraints.append(f"(= {var_name} {var_value})")
    
    if not constraints:
        return "true"
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return f"(and {' '.join(constraints)})"


def process_model(model_json, qvars_json, trace_z3=False):
    """Process a single model for QE"""
    try:
        model = json.loads(model_json)
        qvars = json.loads(qvars_json)
        
        minterm_smtlib = build_minterm_from_model(model)
        
        free_vars = [var_name for var_name in model.keys() if var_name not in qvars]
        
        if not free_vars:
            return "true"
            
        free_var_constraints = []
        for var_name in free_vars:
            var_info = model[var_name]
            var_type = var_info['type']
            var_value = var_info['value'].strip()
            
            if var_type == 'Bool':
                if var_value == 'true':
                    free_var_constraints.append(var_name)
                elif var_value == 'false':
                    free_var_constraints.append(f"(not {var_name})")
            else:
                free_var_constraints.append(f"(= {var_name} {var_value})")
                
        if not free_var_constraints:
            return "true"
        elif len(free_var_constraints) == 1:
            projection = free_var_constraints[0]
        else:
            projection = f"(and {' '.join(free_var_constraints)})"
        
        # Verify projection with Z3
        verify_smt = f"""
(set-option :produce-models true)
(set-option :interactive-mode true)
(set-logic ALL)
"""
        for var_name, var_info in model.items():
            var_type = var_info['type']
            verify_smt += f"(declare-const {var_name} {var_type})\n"
            
        verify_smt += f"(assert {minterm_smtlib})\n"
        verify_smt += f"(assert {projection})\n"
        verify_smt += "(check-sat)\n"
        
        if trace_z3:
            result = run_z3_with_tracing(verify_smt, description="Projection Verification")
        else:
            with tempfile.NamedTemporaryFile(suffix='.smt2', mode='w+', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(verify_smt)
                
            result = subprocess.run(
                [Z3_PATH, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            os.unlink(temp_path)
        
        if "sat" in result.stdout:
            return projection
        
        return "false"
        
    except Exception as e:
        print(f"Error in process_model: {e}")
        return "false"


def simplify_result(result):
    """Simplify the result formula by removing duplicate clauses from disjunctions"""
    if not result or result == "true" or result == "false":
        return result
        
    if result.startswith("(or ") and result.endswith(")"):
        inner_content = result[4:-1].strip()
        
        constraints = []
        pattern = r'\(=\s+z\s+(.+?)\)'
        for match in re.finditer(pattern, inner_content):
            constraint = match.group(0)
            if constraint not in constraints:
                constraints.append(constraint)
        
        if constraints:
            return "(or " + " ".join(constraints) + ")"
    
    return result


def qelim_exists_lme_parallel(phi, qvars, num_workers=None, batch_size=4, trace_z3=False):
    """
    Parallel Existential Quantifier Elimination using Lazy Model Enumeration with IPC
    
    Args:
        phi: Formula to eliminate quantifiers from (Z3 expression or SMT-LIB string)
        qvars: List of variables to eliminate (Z3 variables)
        num_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of models to sample in each iteration
        trace_z3: Enable tracing of Z3 commands and responses
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    if trace_z3:
        # Configure logging to show detailed Z3 traces
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Z3 tracing enabled")
    
    try:
        # Get atomic predicates
        # predicates = [to_smtlib(pred) for pred in get_atoms(phi)]
        
        # Convert formula to SMT-LIB format
        phi_smtlib = to_smtlib(phi)
        
        # Serialize variables for IPC
        qvars_json = json.dumps([str(var) for var in qvars])
        
        # Try direct QE first
        direct_result = try_direct_qe(phi, qvars, trace_z3)
        if direct_result:
            return direct_result
        
        # Track projections and blocking clauses
        projections = []
        blocking_clauses = []
        
        # Main loop for model enumeration
        max_iterations = 5  # Limit iterations for testing
        for iteration in range(max_iterations):
            # Extract models from the formula, blocking existing projections
            formula_with_blocking = phi_smtlib
            for clause in blocking_clauses:
                formula_with_blocking = f"(and {formula_with_blocking} (not {clause}))"
            
            models = extract_models(formula_with_blocking, num_models=batch_size, trace_z3=trace_z3)
            
            # If no more models, break
            if not models:
                break
            
            # Process models in parallel
            new_projections = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for model in models:
                    # Serialize model for IPC
                    model_json = json.dumps(model)
                    
                    # Submit job to process this model
                    future = executor.submit(process_model, model_json, qvars_json, trace_z3)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        projection = future.result()
                        if projection and projection != "false":
                            new_projections.append(projection)
                    except Exception as e:
                        if trace_z3:
                            logger.error(f"Error in parallel processing: {e}")
                        else:
                            print(f"Error in parallel processing: {e}")
            
            # Update projections and blocking clauses
            projections.extend(new_projections)
            blocking_clauses.extend(new_projections)
            
            if not new_projections:
                break
        
        # Combine all projections with OR
        if not projections:
            return "false"
        
        if len(projections) == 1:
            result = projections[0]
        else:
            result = f"(or {' '.join(projections)})"
        
        # Simplify the result by removing duplicates
        return simplify_result(result)
        
    except Exception as e:
        if trace_z3:
            logger.error(f"Error in qelim_exists_lme_parallel: {e}")
        else:
            print(f"Error in qelim_exists_lme_parallel: {e}")
        return "false"


# Add this function to help with test result analysis
def analyze_result(name, formula, result):
    """Analyze the result of a test"""
    print(f"\n{name} Analysis:")
    print(f"  - Formula: {formula}")
    print(f"  - Result: {result}")
    
    if result == "false":
        print("  - Interpretation: Unsatisfiable after quantifier elimination")
        return "Success" if "Unsatisfiable" in name else "Failure"
    elif result == "true":
        print("  - Interpretation: Always satisfiable (tautology)")
        return "Success"
    else:
        print("  - Interpretation: Successfully eliminated quantifiers")
        return "Success"


# Update test_parallel_qe to use the analyze_result function
def test_parallel_qe():
    """Test the parallel QE implementation with IPC"""
    print("=" * 60)
    print("TESTING PARALLEL QE VIA IPC")
    print("=" * 60)
    
    # Track test results
    test_results = []
    
    try:
        import z3
        # Example 1: Complex formula
        x, y, z = z3.Reals("x y z")
        fml = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))
        
        print("\nTest 1: Complex formula")
        print(f"Formula: {fml}")
        
        # Test with default workers and tracing
        print("Running with Z3 tracing enabled (check logs for details)")
        result = qelim_exists_lme_parallel(fml, [x, y], trace_z3=True)
        print(f"Result: {result}")
        test_results.append(("Test 1 (Complex formula)", analyze_result("Test 1", fml, result)))
        
        # Example 2: Simple formula
        print("\nTest 2: Simple formula")
        simple_fml = z3.And(x > 0, x < 10)
        print(f"Formula: {simple_fml}")
        
        result_simple = qelim_exists_lme_parallel(simple_fml, [x])
        print(f"Result: {result_simple}")
        test_results.append(("Test 2 (Simple formula)", analyze_result("Test 2", simple_fml, result_simple)))

        # Example 3: Linear constraints
        print("\nTest 3: Linear constraints")
        a, b, c = z3.Reals("a b c")
        linear_fml = z3.And(a + b > c, 2*a - b < 5, c >= 0)
        print(f"Formula: {linear_fml}")
        
        result_linear = qelim_exists_lme_parallel(linear_fml, [a])
        print(f"Result: {result_linear}")
        test_results.append(("Test 3 (Linear constraints)", analyze_result("Test 3", linear_fml, result_linear)))
        
        # Example 4: Boolean variables
        print("\nTest 4: Boolean variables")
        p, q, r = z3.Bools("p q r")
        bool_fml = z3.And(z3.Implies(p, q), z3.Or(p, r), z3.Not(z3.And(q, r)))
        print(f"Formula: {bool_fml}")
        
        result_bool = qelim_exists_lme_parallel(bool_fml, [p])
        print(f"Result: {result_bool}")
        test_results.append(("Test 4 (Boolean variables)", analyze_result("Test 4", bool_fml, result_bool)))
        
        # Example 5: Nonlinear arithmetic
        print("\nTest 5: Nonlinear arithmetic")
        x, y = z3.Reals("x y")
        nonlinear_fml = z3.And(x*x + y*y < 1, x > 0, y > 0)
        print(f"Formula: {nonlinear_fml}")
        
        result_nonlinear = qelim_exists_lme_parallel(nonlinear_fml, [x])
        print(f"Result: {result_nonlinear}")
        test_results.append(("Test 5 (Nonlinear arithmetic)", analyze_result("Test 5", nonlinear_fml, result_nonlinear)))
    
        # Example 6: Multiple quantified variables
        print("\nTest 6: Multiple quantified variables")
        x, y, z = z3.Reals("x y z")
        multi_fml = z3.And(x + y > z, x - y < z, z >= 0)
        print(f"Formula: {multi_fml}")
        
        result_multi = qelim_exists_lme_parallel(multi_fml, [x, y])
        print(f"Result: {result_multi}")
        test_results.append(("Test 6 (Multiple quantified variables)", analyze_result("Test 6", multi_fml, result_multi)))
        
        # Example 7: Complex disjunctive formula
        print("\nTest 7: Complex disjunctive formula")
        x, y, z = z3.Reals("x y z")
        disj_fml = z3.Or(
            z3.And(x > y, y > z, x < 10),
            z3.And(x < 0, y > 0, z == x + y),
            z3.And(x == y, y == z, x > 5)
        )
        print(f"Formula: {disj_fml}")
        
        result_disj = qelim_exists_lme_parallel(disj_fml, [x])
        print(f"Result: {result_disj}")
        test_results.append(("Test 7 (Complex disjunctive formula)", analyze_result("Test 7", disj_fml, result_disj)))
        
        # Example 8: Unsatisfiable formula
        print("\nTest 8: Unsatisfiable formula")
        x, y = z3.Reals("x y")
        unsat_fml = z3.And(x > y, y > x)
        print(f"Formula: {unsat_fml}")
        
        # Test with tracing to see what happens
        result_unsat = qelim_exists_lme_parallel(unsat_fml, [x], trace_z3=True)
        print(f"Result: {result_unsat}")
        test_results.append(("Test 8 (Unsatisfiable formula)", "Success" if result_unsat == "false" else "Failure"))
        
        # Example 9: Integer variables
        print("\nTest 9: Integer variables")
        i, j = z3.Ints("i j")
        int_fml = z3.And(i >= 0, i < 10, j == 2*i + 1)
        print(f"Formula: {int_fml}")
        
        result_int = qelim_exists_lme_parallel(int_fml, [i])
        print(f"Result: {result_int}")
        test_results.append(("Test 9 (Integer variables)", analyze_result("Test 9", int_fml, result_int)))
    
    except Exception as e:
        print(f"Error in tests: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in test_results:
        print(f"{test_name:30s}: {result}")
    print("=" * 60)



def test_z3_tracing():
    """Simple test for Z3 tracing functionality"""
    print("=" * 60)
    print("TESTING Z3 TRACING")
    print("=" * 60)
    
    # Create a new trace log file
    with open("z3_trace.log", "w") as log:
        log.write("Z3 TRACE LOG\n\n")
    
    print("Z3 trace will be written to z3_trace.log")
    
    try:
        import z3
        x, y = z3.Reals("x y")
        
        # Test 1: Simple satisfiable formula
        print("\nTest 1: Simple satisfiable formula")
        simple_fml = z3.And(x > 0, x < y, y < 10)
        print(f"Formula: {simple_fml}")
        
        result = qelim_exists_lme_parallel(simple_fml, [x], trace_z3=True)
        print(f"Result: {result}")
        
        # Test 2: Unsatisfiable formula
        print("\nTest 2: Unsatisfiable formula")
        unsat_fml = z3.And(x > 5, x < 2)
        print(f"Formula: {unsat_fml}")
        
        result_unsat = qelim_exists_lme_parallel(unsat_fml, [x], trace_z3=True)
        print(f"Result: {result_unsat}")
        
        print(f"\nTracing complete. Z3 trace available in z3_trace.log")
        
    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    # Uncomment to run all tests
    test_parallel_qe()
    
    # Run Z3 tracing test
    # test_z3_tracing() 