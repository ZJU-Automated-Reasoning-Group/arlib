"""
TODO: by LLM, to check.

Implementation of MaxSMT solving algorithms

This module provides various algorithms for solving MaxSMT problems:
1. Core-guided approach (Fu-Malik / MSUS3) - SAT'13
2. Implicit Hitting Set approach (IHS) - IJCAR'18 
3. Local search for MaxSMT(LIA)

MaxSMT extends MaxSAT to SMT formulas, where each formula can have a weight 
(hard constraints have infinite weight, soft constraints have finite weights).
The goal is to find an assignment that maximizes the sum of weights of satisfied constraints.
"""

import time
import logging
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from enum import Enum

import z3

logger = logging.getLogger(__name__)


class MaxSMTAlgorithm(Enum):
    """Enumeration of available MaxSMT algorithms"""
    CORE_GUIDED = "core-guided"   # Core-guided algorithm from SAT'13
    IHS = "ihs"                   # Implicit Hitting Set algorithm from IJCAR'18
    LOCAL_SEARCH = "local-search"  # Local search for MaxSMT(LIA)


class MaxSMTSolver:
    """
    Main class for solving MaxSMT problems
    """
    def __init__(self, algorithm: str = "core-guided", solver_name: str = "z3"):
        """Initialize the MaxSMT solver

        Args:
            algorithm: Algorithm to use (core-guided, ihs, local-search, z3-opt)
            solver_name: Name of the underlying SMT solver
        """
        self.algorithm = algorithm
        if algorithm != "z3-opt":
            self.algorithm = MaxSMTAlgorithm(algorithm)
        self.solver_name = solver_name
        self.hard_constraints = []
        self.soft_constraints = []
        self.weights = []
        self.solver = None
        self._setup_solver(solver_name)

    def _setup_solver(self, solver_name: str):
        """Set up the underlying SMT solver
        
        Args:
            solver_name: Name of the SMT solver to use
        """
        if solver_name == "z3":
            self.solver = z3.Solver()
        else:
            # For other solvers, we could add adapter code here
            raise NotImplementedError(f"Solver {solver_name} not supported")

    def add_hard_constraint(self, constraint):
        """Add a hard constraint (must be satisfied)
        
        Args:
            constraint: SMT formula that must be satisfied
        """
        self.hard_constraints.append(constraint)
    
    def add_hard_constraints(self, constraints):
        """Add multiple hard constraints
        
        Args:
            constraints: List of SMT formulas that must be satisfied
        """
        self.hard_constraints.extend(constraints)

    def add_soft_constraint(self, constraint, weight: float = 1.0):
        """Add a soft constraint with a weight
        
        Args:
            constraint: SMT formula that should be satisfied if possible
            weight: Weight of the constraint (higher = more important)
        """
        self.soft_constraints.append(constraint)
        self.weights.append(weight)
    
    def add_soft_constraints(self, constraints, weights=None):
        """Add multiple soft constraints with weights
        
        Args:
            constraints: List of SMT formulas 
            weights: List of weights (default: all 1.0)
        """
        if weights is None:
            weights = [1.0] * len(constraints)
        
        if len(constraints) != len(weights):
            raise ValueError("Number of constraints must match number of weights")
        
        self.soft_constraints.extend(constraints)
        self.weights.extend(weights)
    
    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Solve the MaxSMT problem using the selected algorithm
        
        Returns:
            Tuple of (sat, model, optimal_cost)
            sat: True if a solution was found
            model: z3 model if sat is True, None otherwise
            optimal_cost: Sum of weights of unsatisfied soft constraints
        """
        if self.algorithm == "z3-opt":
            return self._solve_z3_optimize()
        elif self.algorithm == MaxSMTAlgorithm.CORE_GUIDED:
            return self._solve_core_guided()
        elif self.algorithm == MaxSMTAlgorithm.IHS:
            return self._solve_ihs()
        elif self.algorithm == MaxSMTAlgorithm.LOCAL_SEARCH:
            return self._solve_local_search()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _solve_z3_optimize(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Solve the MaxSMT problem using Z3's Optimize engine
        
        Z3's Optimize engine can directly handle weighted MaxSMT problems.
        This method uses Z3's built-in functionality.
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Create Z3 Optimize solver
        opt = z3.Optimize()
        
        # Add hard constraints
        for hc in self.hard_constraints:
            opt.add(hc)
        
        # Create relaxation variables for soft constraints
        relax_vars = []
        for i, (sc, weight) in enumerate(zip(self.soft_constraints, self.weights)):
            # Create a relaxation variable for this soft constraint
            b = z3.Bool(f"_soft_{i}")
            
            # Add the implication: if b is true, then the soft constraint must hold
            opt.add(z3.Implies(b, sc))
            
            # Add to the list for the objective function
            relax_vars.append((b, weight))
        
        # Set the objective: maximize the sum of weights of satisfied soft constraints
        objective = opt.maximize(z3.Sum([z3.If(b, weight, 0) for b, weight in relax_vars]))
        
        # Check if the formula is satisfiable
        if opt.check() == z3.sat:
            model = opt.model()
            
            # Calculate the cost (sum of weights of violated constraints)
            total_weight = sum(self.weights)
            
            # Properly handle the objective value
            try:
                # Get the objective value
                obj_val = opt.upper(objective)
                
                # Convert to float - different Z3 versions may return different types
                if hasattr(obj_val, 'as_decimal'):
                    satisfied_weight = float(obj_val.as_decimal(10).strip('?'))
                elif hasattr(obj_val, 'as_fraction'):
                    fraction = obj_val.as_fraction()
                    satisfied_weight = float(fraction.numerator) / float(fraction.denominator)
                elif hasattr(obj_val, 'as_long'):
                    satisfied_weight = float(obj_val.as_long())
                elif hasattr(obj_val, 'as_float'):
                    satisfied_weight = obj_val.as_float()
                else:
                    # Last resort: convert to string and parse
                    satisfied_weight = float(str(obj_val).replace('?', ''))
            except (ValueError, AttributeError):
                # Alternative method: evaluate the objective expression in the model
                satisfied_weight = 0.0
                for (b, weight) in relax_vars:
                    if z3.is_true(model.eval(b)):
                        satisfied_weight += weight
            
            # Cost is the weight of violated constraints
            cost = total_weight - satisfied_weight
            
            return True, model, cost
        else:
            # Formula is unsatisfiable
            return False, None, float('inf')

    def _solve_core_guided(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Core-guided algorithm for MaxSMT (Fu-Malik/MSUS3 variant)
        
        Relaxes soft constraints by adding relaxation variables and uses
        the SMT solver to find unsatisfiable cores iteratively.
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Create relaxation variables for soft constraints
        relax_vars = [z3.Bool(f"_relax_{i}") for i in range(len(self.soft_constraints))]
        
        # Add hard constraints to solver
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)
        
        # Check if hard constraints are satisfiable
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')
        
        # Initialize with all soft constraints with relaxation variables
        relaxed_soft = []
        for i, sc in enumerate(self.soft_constraints):
            # Add relaxed version of soft constraint: sc OR relax_var
            relaxed_soft.append(z3.Or(sc, relax_vars[i]))
            solver.add(z3.Or(sc, relax_vars[i]))
        
        # Keep track of blocking variables and their weights
        block_vars = []
        block_weights = []
        
        # Main loop: find and relax cores
        while True:
            # Assume all relaxation variables are false (soft constraints must be satisfied)
            assumptions = [z3.Not(rv) for rv in relax_vars if rv not in block_vars]
            
            # Check satisfiability with current assumptions
            result = solver.check(assumptions)
            
            if result == z3.sat:
                # All remaining soft constraints can be satisfied
                model = solver.model()
                
                # Calculate cost: sum of weights of unsatisfied soft constraints
                cost = sum(self.weights[i] for i in range(len(self.soft_constraints)) 
                          if model.evaluate(relax_vars[i], model_completion=True))
                
                # Add cost from blocked vars
                cost += sum(w for w in block_weights)
                
                return True, model, cost
            
            # Get the unsatisfiable core
            core = solver.unsat_core()
            
            if not core:
                # No core found, problem is unsatisfiable
                return False, None, float('inf')
            
            # Find soft constraints in the core
            core_indices = []
            for i, rv in enumerate(relax_vars):
                if z3.Not(rv) in core:
                    core_indices.append(i)
            
            if not core_indices:
                # No soft constraints in the core, problem is unsatisfiable
                return False, None, float('inf')
            
            # Find the minimum weight in the core
            min_weight = min(self.weights[i] for i in core_indices)
            
            # Create a blocking variable for this core
            block_var = z3.Bool(f"_block_{len(block_vars)}")
            block_vars.append(block_var)
            block_weights.append(min_weight)
            
            # Add cardinality constraint to allow at most one soft constraint to be relaxed in the core
            at_most_one = []
            for i in core_indices:
                # Update the weight by subtracting the minimum weight
                self.weights[i] -= min_weight
                
                # If weight becomes 0, move to block_vars
                if self.weights[i] < 1e-6:  # Numerical tolerance
                    if relax_vars[i] not in block_vars:
                        block_vars.append(relax_vars[i])
                
                # Add relaxation variable to at_most_one constraint
                at_most_one.append(relax_vars[i])
            
            # Add constraint: block_var OR at_most_one(relax_vars in core)
            at_most_one_constraint = z3.PbLe([(var, 1) for var in at_most_one], 1)
            solver.add(z3.Or(block_var, at_most_one_constraint))

    def _solve_ihs(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Implicit Hitting Set algorithm for MaxSMT
        
        Iteratively finds optimal hitting sets for the collection of cores.
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)
        
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')
        
        # Create relaxation variables for soft constraints
        relax_vars = [z3.Bool(f"_relax_{i}") for i in range(len(self.soft_constraints))]
        
        # Solver for finding cores
        core_solver = z3.Solver()
        
        # Add hard constraints
        for hc in self.hard_constraints:
            core_solver.add(hc)
        
        # Add soft constraints with relaxation variables
        for i, sc in enumerate(self.soft_constraints):
            core_solver.add(z3.Or(sc, relax_vars[i]))
        
        # Set to keep track of all discovered cores
        all_cores = []
        
        # Current best model and cost
        best_model = None
        best_cost = float('inf')
        
        # Solver for the hitting set problem
        hs_solver = z3.Optimize()
        
        # Add variables for the hitting set problem
        hs_vars = [z3.Bool(f"_hs_{i}") for i in range(len(self.soft_constraints))]
        
        # Add objective: minimize sum of weights of selected soft constraints
        obj_terms = [(hs_vars[i], self.weights[i]) for i in range(len(self.soft_constraints))]
        hs_solver.minimize(z3.Sum([z3.If(var, weight, 0) for var, weight in obj_terms]))
        
        while True:
            # Solve the hitting set problem
            if hs_solver.check() == z3.sat:
                hs_model = hs_solver.model()
                hitting_set = [i for i in range(len(self.soft_constraints)) 
                              if hs_model.evaluate(hs_vars[i])]
                
                current_cost = sum(self.weights[i] for i in hitting_set)
                
                # If the current cost equals the best cost, we're done
                if best_model is not None and abs(current_cost - best_cost) < 1e-6:
                    return True, best_model, best_cost
                
                # Check if the current hitting set gives a satisfiable formula
                assumptions = [z3.Not(relax_vars[i]) for i in range(len(self.soft_constraints)) 
                              if i not in hitting_set]
                
                # Check satisfiability with assumptions
                result = core_solver.check(assumptions)
                
                if result == z3.sat:
                    # Found a better solution
                    best_model = core_solver.model()
                    best_cost = current_cost
                    
                    # Add constraint to find a better solution
                    # Fix: Use boolean literals instead of PbLe with float constant
                    # Create a constraint that says "find a solution with strictly less cost"
                    # by explicitly adding up the variables we want to minimize
                    if hitting_set:  # If there are soft constraints violated
                        # We want to find a solution where fewer than current_violated soft constraints are violated
                        # Instead of using PbLe with float, use a logical constraint
                        # Exclude at least one of the current violated constraints
                        hs_solver.add(z3.Not(z3.And([hs_vars[i] for i in hitting_set])))
                    else:
                        # If no soft constraints are violated, we've found the optimal solution
                        return True, best_model, best_cost
                else:
                    # Extract the unsatisfiable core
                    core = core_solver.unsat_core()
                    core_indices = [i for i in range(len(self.soft_constraints)) 
                                   if z3.Not(relax_vars[i]) in core]
                    
                    if not core_indices:
                        # No soft constraints in the core
                        if best_model is not None:
                            return True, best_model, best_cost
                        else:
                            return False, None, float('inf')
                    
                    # Add the core to the set of all cores
                    all_cores.append(core_indices)
                    
                    # Add constraint: at least one variable from the core must be hit
                    hs_solver.add(z3.Or([hs_vars[i] for i in core_indices]))
            else:
                # Hitting set problem is unsatisfiable
                if best_model is not None:
                    return True, best_model, best_cost
                else:
                    return False, None, float('inf')

    def _solve_local_search(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Local search algorithm for MaxSMT(LIA)
        
        Uses local search techniques to find a solution.
        Currently only implemented for SMT formulas over linear integer arithmetic.
        
        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # This is a simplified implementation of local search for MaxSMT
        # A full implementation would require more extensive code
        
        # First check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)
        
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')
        
        # Start with a model that satisfies hard constraints
        current_model = solver.model()
        
        # Get all variables in the formula
        all_vars = set()
        for hc in self.hard_constraints:
            all_vars.update(self._get_variables(hc))
        for sc in self.soft_constraints:
            all_vars.update(self._get_variables(sc))
        
        # Convert the set to a list for iteration
        all_vars = list(all_vars)
        
        # Initialize best model and cost
        best_model = current_model
        best_cost = self._calculate_cost(current_model)
        
        # Maximum number of iterations
        max_iterations = 1000
        
        # Main local search loop
        for _ in range(max_iterations):
            # Try to improve the current model
            improved = False
            
            # Try to modify each variable
            for var in all_vars:
                if not z3.is_int(var):
                    continue  # Currently only supports integer variables
                
                try:
                    # Get current value
                    current_value = current_model.eval(var, model_completion=True).as_long()
                    
                    # Try neighboring values
                    for offset in [-2, -1, 1, 2]:
                        new_value = current_value + offset
                        
                        # Create a temporary solver and assign the new value
                        temp_solver = z3.Solver()
                        
                        # Add all hard constraints
                        for hc in self.hard_constraints:
                            temp_solver.add(hc)
                        
                        # Add the fixed value for this variable
                        temp_solver.add(var == new_value)
                        
                        # Add the values for all other variables from the current model
                        # (except the one we're modifying)
                        for other_var in all_vars:
                            if other_var != var and other_var in current_model.decls():
                                val = current_model.eval(other_var, model_completion=True)
                                temp_solver.add(other_var == val)
                        
                        # Check if satisfiable
                        if temp_solver.check() == z3.sat:
                            # Get the new model
                            new_model = temp_solver.model()
                            
                            # Calculate the new cost
                            new_cost = self._calculate_cost(new_model)
                            
                            # If better than current best, update
                            if new_cost < best_cost:
                                best_model = new_model
                                best_cost = new_cost
                                current_model = new_model
                                improved = True
                                break
                except:
                    # Skip if there are any errors (e.g., variable not in model)
                    continue
                
                if improved:
                    break
            
            # If no improvement was found, stop
            if not improved:
                break
        
        return True, best_model, best_cost
            
    def _get_variables(self, formula):
        """Extract variables from a Z3 formula"""
        variables = set()
        
        def collect(expr):
            if z3.is_const(expr) and not z3.is_bool(expr) and not z3.is_true(expr) and not z3.is_false(expr):
                variables.add(expr)
            else:
                for child in expr.children():
                    collect(child)
        
        collect(formula)
        return variables
    
    def _calculate_cost(self, model):
        """Calculate the cost of a model (sum of weights of unsatisfied soft constraints)"""
        cost = 0.0
        for i, sc in enumerate(self.soft_constraints):
            if not self._evaluate(model, sc):
                cost += self.weights[i]
        return cost
    
    def _evaluate(self, model, formula):
        """Evaluate a formula under a model"""
        try:
            return z3.is_true(model.eval(formula, model_completion=True))
        except:
            # For complex formulas that can't be directly evaluated
            return False
    
    def _satisfies_hard_constraints(self, model):
        """Check if a model satisfies all hard constraints"""
        for hc in self.hard_constraints:
            if not self._evaluate(model, hc):
                return False
        return True
    
    def _create_modified_model(self, model, var, new_value):
        """Create a new model with a modified variable value"""
        # NOTE: This method is no longer used as we create temporary solvers
        # directly in the _solve_local_search method
        
        # Create a new solver
        solver = z3.Solver()
        
        # Add constraints for all variables in the model
        for decl in model.decls():
            v = decl()
            # Skip the variable we want to modify
            if v != var:
                solver.add(v == model[decl])
        
        # Add constraint for the modified variable
        solver.add(var == new_value)
        
        # Check if satisfiable
        if solver.check() == z3.sat:
            return solver.model()
        
        # If not satisfiable, return the original model
        return model


def solve_maxsmt(hard_constraints: List[z3.ExprRef],
                 soft_constraints: List[z3.ExprRef],
                 weights: List[float] = None,
                 algorithm: str = "core-guided",
                 solver_name: str = "z3") -> Tuple[bool, Optional[z3.ModelRef], float]:
    """Convenience function to solve a MaxSMT problem
    
    Args:
        hard_constraints: List of hard constraints (must be satisfied)
        soft_constraints: List of soft constraints (may be violated)
        weights: List of weights for soft constraints (default: all 1.0)
        algorithm: Algorithm to use (core-guided, ihs, local-search, z3-opt)
        solver_name: Underlying SMT solver to use
    
    Returns:
        Tuple of (sat, model, optimal_cost)
        sat: True if a solution was found
        model: z3 model if sat is True, None otherwise
        optimal_cost: Sum of weights of unsatisfied soft constraints
    """
    if weights is None:
        weights = [1.0] * len(soft_constraints)
    
    solver = MaxSMTSolver(algorithm, solver_name)
    solver.add_hard_constraints(hard_constraints)
    solver.add_soft_constraints(soft_constraints, weights)
    
    return solver.solve()


def demo():
    """Demonstrate the MaxSMT solver with a simple example"""
    import time
    
    # Create variables
    x, y = z3.Ints('x y')
    
    # Hard constraints
    hard = [x >= 0, y >= 0, x + y <= 10]
    
    # Soft constraints with weights
    soft = [x == 5, y == 5, x + y == 8]
    weights = [1.0, 1.0, 2.0]
    
    # Solve using different algorithms
    algorithms = ["core-guided", "ihs", "local-search", "z3-opt"]
    
    for alg in algorithms:
        print(f"\nSolving with {alg}")
        start_time = time.time()
        sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm=alg)
        end_time = time.time()
        
        if sat:
            print(f"Model: x = {model.eval(x)}, y = {model.eval(y)}")
            print(f"Cost: {cost}")
        else:
            print("Unsatisfiable")
        
        print(f"Time: {end_time - start_time:.4f} seconds")


def example_scheduling():
    """Example: Job scheduling problem with MaxSMT
    
    We have jobs with durations and deadlines.
    Hard constraints: Jobs cannot overlap
    Soft constraints: Try to meet deadlines
    """
    print("\n=== Job Scheduling Example ===")
    
    # Number of jobs
    n_jobs = 5
    
    # Job durations (time units)
    durations = [3, 2, 4, 1, 5]
    
    # Job deadlines
    deadlines = [5, 4, 10, 3, 12]
    
    # Create variables for start times
    starts = [z3.Int(f"start_{i}") for i in range(n_jobs)]
    
    # Hard constraints
    hard = []
    
    # All jobs must start at or after time 0
    for i in range(n_jobs):
        hard.append(starts[i] >= 0)
    
    # Jobs cannot overlap (for each pair of jobs)
    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
            # Either job i finishes before job j starts, or vice versa
            hard.append(z3.Or(
                starts[i] + durations[i] <= starts[j],  # job i before job j
                starts[j] + durations[j] <= starts[i]   # job j before job i
            ))
    
    # Soft constraints: try to meet deadlines
    soft = []
    weights = []
    
    for i in range(n_jobs):
        # Try to finish before deadline
        soft.append(starts[i] + durations[i] <= deadlines[i])
        # Assign weights based on job importance (for this example, all equal)
        weights.append(1.0)
    
    # Solve using different algorithms
    algorithms = ["core-guided", "z3-opt"]
    all_results = {}
    
    for alg in algorithms:
        print(f"\nSolving scheduling problem with {alg}")
        start_time = time.time()
        sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm=alg)
        end_time = time.time()
        
        if sat:
            print("Optimal Schedule:")
            jobs_info = [(i, model.eval(starts[i]).as_long(), 
                         model.eval(starts[i]).as_long() + durations[i],
                         deadlines[i]) for i in range(n_jobs)]
            
            # Store results for comparison
            all_results[alg] = {
                'cost': cost,
                'jobs_info': jobs_info
            }
            
            # Sort by start time
            jobs_info.sort(key=lambda x: x[1])
            
            print("Job  Start  End  Deadline  Status")
            print("----  -----  ---  --------  ------")
            missed = 0
            for job, start, end, deadline in jobs_info:
                status = "On time" if end <= deadline else "Late"
                if status == "Late":
                    missed += 1
                print(f"{job:3}   {start:4}  {end:3}   {deadline:6}    {status}")
                
            print(f"\nMissed deadlines: {missed}")
            print(f"Cost: {cost}")
        else:
            print("Unsatisfiable")
        
        print(f"Time: {end_time - start_time:.4f} seconds")
    
    # Compare results if we have both algorithms
    if len(all_results) == 2:
        print("\n=== Comparing Results Between Algorithms ===")
        
        # Extract results for each algorithm
        cg_results = all_results.get("core-guided")
        z3_results = all_results.get("z3-opt")
        
        if cg_results and z3_results:
            print(f"Core-guided cost: {cg_results['cost']}, Z3-Opt cost: {z3_results['cost']}")
            
            if abs(cg_results['cost'] - z3_results['cost']) < 1e-6:
                print("Both algorithms found solutions with the same cost.")
            else:
                print("The algorithms found solutions with different costs!")
                
                # Which constraints are violated in each solution?
                print("\nSoft constraints violated in each solution:")
                print("Constraint | Core-guided | Z3-Opt")
                print("-----------+-------------+-------")
                
                # Get job results by job ID for easier comparison
                cg_jobs = {job[0]: job for job in cg_results['jobs_info']}
                z3_jobs = {job[0]: job for job in z3_results['jobs_info']}
                
                for i in range(n_jobs):
                    cg_job = cg_jobs[i]
                    z3_job = z3_jobs[i]
                    
                    cg_end = cg_job[2]
                    z3_end = z3_job[2]
                    deadline = deadlines[i]
                    
                    cg_violation = "Late" if cg_end > deadline else "On time"
                    z3_violation = "Late" if z3_end > deadline else "On time"
                    
                    print(f"Job {i:3}    | {cg_violation:11} | {z3_violation}")
                
                print("\nDetailed job comparison:")
                print("Job | Core-guided (Start-End) | Z3-Opt (Start-End) | Deadline")
                print("----+------------------------+------------------+--------")
                
                for i in range(n_jobs):
                    cg_job = cg_jobs[i]
                    z3_job = z3_jobs[i]
                    
                    cg_start = cg_job[1]
                    cg_end = cg_job[2]
                    z3_start = z3_job[1]
                    z3_end = z3_job[2]
                    deadline = deadlines[i]
                    
                    print(f"{i:3} | {cg_start:2}-{cg_end:2} {' (Late)' if cg_end > deadline else '       '} | "
                          f"{z3_start:2}-{z3_end:2} {' (Late)' if z3_end > deadline else '       '} | {deadline}")


def example_minimal_correction():
    """Example: Minimal correction subset
    
    Find a minimal subset of clauses to remove to make an unsatisfiable formula satisfiable.
    This is the dual of the MaxSAT problem and is useful in debugging.
    """
    print("\n=== Minimal Correction Subset Example ===")
    
    # Create variables
    a, b, c, d = z3.Bools('a b c d')
    
    # Create a set of clauses that is unsatisfiable
    # (a AND b AND NOT a) is unsatisfiable
    clauses = [
        a,             # clause 1
        b,             # clause 2
        z3.Not(a),     # clause 3
        z3.Implies(b, c),  # clause 4
        z3.Implies(c, d)   # clause 5
    ]
    
    # No hard constraints
    hard = []
    
    # Each clause is a soft constraint
    soft = clauses
    
    # All clauses have equal weight
    weights = [1.0] * len(clauses)
    
    # Solve using Z3's optimize engine
    print("\nFinding minimal correction subset")
    sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm="z3-opt")
    
    if sat:
        print("\nSolution found by removing these clauses:")
        
        for i, clause in enumerate(clauses):
            if not model.evaluate(clause):
                print(f"Clause {i+1}: {clause}")
        
        print(f"\nCost (number of clauses removed): {cost}")
        print(f"Model: a={model.eval(a)}, b={model.eval(b)}, c={model.eval(c)}, d={model.eval(d)}")
    else:
        print("Unexpected: Could not find a solution")


if __name__ == '__main__':
    print("\n=== Basic Example ===")
    demo()
    example_scheduling()
    example_minimal_correction()
