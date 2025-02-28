# coding: utf-8
"""
A simple/native implementation of DPLL algorithm
"""
import copy
from typing import List, Tuple, Dict, Optional, Set, Union

Assignment = Dict[int, bool]
Formula = List[List[int]]
Result = Union[str, Assignment]


def is_unit(clause: List[int]) -> bool:
    return len(clause) == 1


def unit_propagation(s: Formula) -> Tuple[Optional[Dict[int, bool]], Formula]:
    """
    Perform unit propagation on the given formula s.

    Args:
        s: A list of lists representing a CNF formula.
    Returns:
        A tuple containing (assignment dict, simplified formula) or (None, simplified formula)
    """
    assignments: Dict[int, bool] = {}
    already_propagated = set()
    i = 0
    
    while i < len(s):
        new_s = []
        clause = s[i]
        
        if is_unit(clause) and tuple(clause) not in already_propagated:
            literal = clause[0]
            assignments[abs(literal)] = literal > 0
            new_s.append(clause)
            already_propagated.add(tuple(clause))
            
            for other_clause in s:
                if -literal in other_clause:
                    new_clause = [lit for lit in other_clause if lit != -literal]
                    if not new_clause:  # Empty clause means contradiction
                        return None, [[]]
                    new_s.append(new_clause)
                elif literal not in other_clause:
                    new_s.append(other_clause)
            
            s = new_s
            i = 0
        else:
            i += 1
    
    for clause in s:
        if not is_unit(clause):
            return None, s
    
    return assignments, []


def atomic_cut(s: Formula) -> List[Formula]:
    """Choose a variable for branching using a simple heuristic."""
    if not s:
        return [s]
        
    # Choose the shortest non-unit clause for branching
    non_unit_clauses = [c for c in s if len(c) > 1]
    if not non_unit_clauses:
        return [copy.deepcopy(s)]
        
    clause = min(non_unit_clauses, key=len)
    atom = abs(clause[0])  # Use absolute value for consistency
    
    left_branch = copy.deepcopy(s)
    right_branch = copy.deepcopy(s)
    left_branch.append([atom])
    right_branch.append([-atom])
    return [left_branch, right_branch]


def pure_literal_elimination(s: Formula) -> Formula:
    """
    Eliminate pure literals from the formula.
    A literal is pure if its negation doesn't appear in the formula.
    """
    literals: Set[int] = set()
    for clause in s:
        literals.update(clause)
    
    pure_literals = [lit for lit in literals if -lit not in literals]
    if not pure_literals:
        return s
        
    new_s = []
    for literal in pure_literals:
        new_s = [clause for clause in s if literal not in clause]
        new_s.append([literal])
        s = new_s
    return new_s


def prove(s: Formula) -> Result:
    """
    Prove satisfiability using DPLL algorithm.
    
    Args:
        s: CNF formula as a list of clauses
    Returns:
        Assignment dictionary if satisfiable, "unsatisfiable" otherwise
    """
    # First apply unit propagation
    assignments, clauses = unit_propagation(s)
    
    if [] in clauses:
        return "unsatisfiable"
    if not clauses:
        return assignments if assignments else {}
        
    # Apply pure literal elimination
    clauses = pure_literal_elimination(clauses)
    
    # Branch on a variable
    branches = atomic_cut(clauses)
    for branch in branches:
        answer = prove(branch)
        if answer != "unsatisfiable":
            return answer
            
    return "unsatisfiable"


def sat_solve(cls: Formula) -> Result:
    """
    Solve SAT instance using DPLL algorithm.
    
    Args:
        cls: List of clauses in CNF form
    Returns:
        Assignment dictionary if satisfiable, "unsatisfiable" otherwise
    """
    return prove(cls)


def sat_solve_str_clauses(cls: str) -> Result:
    """
    Solve SAT instance from string representation.
    
    Args:
        cls: String representation of clauses, e.g., "[[1, 2], [-2, 1]]"
    Returns:
        Assignment dictionary if satisfiable, "unsatisfiable" otherwise
    """
    import ast
    return prove(ast.literal_eval(cls))


def test() -> None:
    """Test the SAT solver with some basic examples."""
    test_cases = [
        ([[1, 2], [-1, 2], [-2]], "satisfiable"),
        ([[1], [-1]], "unsatisfiable"),
        ([[1, 2], [-1, 2], [1, -2], [-1, -2]], "satisfiable"),
    ]
    
    for formula, expected in test_cases:
        result = sat_solve(formula)
        status = "satisfiable" if result != "unsatisfiable" else "unsatisfiable"
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        print(f"Status: {'✓' if status == expected else '✗'}\n")


if __name__ == "__main__":
    test()
