"""Parallel Quantifier Elimination via Lazy Model Enumeration (LME-QE)

An implementation of LME-QE (CAV 2013) with parallelized model enumeration and projection.
This algorithm eliminates existential quantifiers through iterative model enumeration
in parallel, potentially improving performance on multi-core systems.
"""

from typing import List, Set
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import z3

from arlib.utils.z3_expr_utils import negate, get_atoms
from arlib.quant.qe.qe_lme import eval_predicates


def process_model(phi, qvars, preds, shared_models):
    """Worker function to process a single model"""
    s = z3.Solver()
    s.add(phi)

    # Block already found models
    for model in shared_models:
        s.add(negate(model))

    if s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        qe_for_conjunction = z3.Tactic('qe2')
        proj = qe_for_conjunction(z3.Exists(qvars, minterm)).as_expr()
        return proj
    return None


def sample_models(phi, num_models, blocked_projections):
    """Sample multiple models from the formula, blocking already found projections"""
    models = []
    s = z3.Solver()
    s.add(phi)
    
    # Block already found projections
    for proj in blocked_projections:
        s.add(negate(proj))
    
    # Try to get up to num_models distinct models
    for _ in range(num_models):
        if s.check() == z3.sat:
            m = s.model()
            models.append(m)
            
            # Add a blocking clause to get a different model next time
            # Create a constraint that blocks this exact model
            block = []
            for d in m:
                if d.arity() > 0:
                    continue
                v = m[d]
                if z3.is_true(v):
                    block.append(d() != v)
                else:
                    block.append(d() != v)
            if block:
                s.add(z3.Or(block))
        else:
            break
            
    return models


# Module-level function that can be pickled for parallel processing
def apply_qe_to_minterm(minterm, qvars):
    """Apply quantifier elimination to a minterm"""
    qe_for_conjunction = z3.Tactic('qe2')
    return qe_for_conjunction(z3.Exists(qvars, minterm)).as_expr()


def qelim_exists_lme_parallel(phi, qvars, num_workers=None, batch_size=4):
    """
    Parallel Existential Quantifier Elimination using Lazy Model Enumeration
    
    Args:
        phi: Formula to eliminate quantifiers from
        qvars: List of variables to eliminate
        num_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of models to sample in each iteration
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    preds = get_atoms(phi)
    projections = set()
    blocked_projections = set()
    
    while True:
        # Sample multiple models
        models = sample_models(phi, batch_size, blocked_projections)
        if not models:
            break
        
        # Process models in parallel
        tasks = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for m in models:
                # Create minterm from current model
                minterm = z3.And(eval_predicates(m, preds))
                
                # Submit projection task to the executor using the picklable function
                task = executor.submit(apply_qe_to_minterm, minterm, qvars)
                tasks.append(task)
        
        # Collect results and update blocked projections
        new_projections = [task.result() for task in tasks]
        projections.update(new_projections)
        blocked_projections.update(new_projections)
    
    return z3.simplify(z3.Or(list(projections)))


def test_parallel_qe():
    """Test the parallel QE implementation"""
    x, y, z = z3.Reals("x y z")
    fml = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))
    
    # Test with different numbers of workers
    qf1 = qelim_exists_lme_parallel(fml, [x, y], num_workers=1)
    print("Result with 1 worker:", qf1)
    
    qf2 = qelim_exists_lme_parallel(fml, [x, y])
    print("Result with default workers:", qf2)
    
    # Verify equivalence with sequential version
    from arlib.quant.qe.qe_lme import qelim_exists_lme
    qf_seq = qelim_exists_lme(fml, [x, y])
    print("Result from sequential algorithm:", qf_seq)
    print("Equivalent:", z3.prove(qf2 == qf_seq))


if __name__ == "__main__":
    test_parallel_qe() 