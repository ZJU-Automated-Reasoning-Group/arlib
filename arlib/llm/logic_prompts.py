"""Advanced prompts for logical reasoning tasks using techniques like:
- Chain of Thought (CoT)
- Self-reflection
- Tree of Thoughts (ToT)
- Step-by-step verification
"""

from typing import Dict, List

class LogicPrompts:
    @staticmethod
    def sat_checking_cot(formula: str, logic: str) -> str:
        return f"""Let's solve this satisfiability checking problem step by step.

Given formula in {logic}:
{formula}

1) First, let's understand what makes a formula satisfiable:
   - A formula is satisfiable if there exists an assignment to variables that makes it true
   - We need to find at least one such assignment, or prove none exists

2) Let's analyze the formula:
   - Identify the variables
   - Understand the constraints
   - Look for obvious contradictions

3) Try to construct a satisfying assignment:
   - Start with simpler subformulas
   - Propagate the constraints
   - Check if assignments are consistent

4) Verify the solution:
   - Substitute the values back
   - Check all constraints are met

Based on this analysis, is the formula satisfiable? 
If yes, provide a satisfying assignment.
If no, explain why no satisfying assignment exists."""

    @staticmethod
    def abduction_self_reflect(background: str, observation: str, variables: List[str]) -> str:
        return f"""Let's find an abductive explanation through careful reasoning and self-reflection.

GIVEN:
Background knowledge: {background}
Observation to explain: {observation}
Available variables: {', '.join(variables)}

REASONING PROCESS:
1) Initial Analysis:
   - What do we know from the background knowledge?
   - What are we trying to explain?
   - What variables are relevant?

2) Generate Candidate Explanations:
   - List possible hypotheses using available variables
   - Order them by simplicity
   - Consider their plausibility

3) Self-Reflection:
   - For each candidate explanation:
     * Does it logically entail the observation when combined with background?
     * Is it the simplest possible explanation?
     * Could there be simpler alternatives?
     * Am I making any unjustified assumptions?

4) Verification:
   - Check logical consistency
   - Verify minimality
   - Ensure all variables used are from the allowed set

Please provide the simplest explanation that passes all these checks."""

    @staticmethod
    def quantifier_elimination_tot(formula: str) -> str:
        return f"""Let's eliminate quantifiers from this formula using Tree of Thoughts approach.

FORMULA: {formula}

Let's explore multiple solution paths in parallel:

PATH 1: Direct Elimination
- Attempt straightforward quantifier elimination
- Consider order of elimination
- Check for simplification opportunities

PATH 2: Variable Substitution
- Identify substitutable variables
- Consider impact on formula structure
- Evaluate resulting complexity

PATH 3: Logical Transformations
- Apply equivalence rules
- Move quantifiers inward
- Simplify subformulas

For each path:
1) Evaluate intermediate results
2) Consider computational cost
3) Check correctness
4) Compare with other paths

Choose the most promising path at each step.
Show work for the best path found.
Verify the final quantifier-free formula is equivalent to the original."""

    @staticmethod
    def interpolant_verification(formula_a: str, formula_b: str) -> str:
        return f"""Let's find and verify a Craig interpolant through systematic reasoning.

FORMULAS:
A: {formula_a}
B: {formula_b}

SYSTEMATIC APPROACH:

1) Variable Analysis:
   - List variables in A: {{vars_a}}
   - List variables in B: {{vars_b}}
   - Identify shared variables: {{shared}}

2) Interpolant Construction:
   - Start with simpler subformulas
   - Use only shared variables
   - Build up complexity gradually

3) Formal Verification:
   Let I be our candidate interpolant.
   Verify:
   a) A → I (A implies I)
   b) I ∧ B is unsatisfiable
   c) I only uses shared variables

4) Simplification:
   - Can the interpolant be made simpler?
   - Are all terms necessary?
   - Check minimality

Provide the interpolant and proof of its correctness."""

    @staticmethod
    def optimization_structured(formula: str, objective: str, minimize: bool) -> str:
        return f"""Let's solve this optimization problem using structured reasoning.

PROBLEM:
Constraints: {formula}
Objective: {objective}
Goal: {"Minimize" if minimize else "Maximize"}

SOLUTION STRATEGY:

1) Problem Analysis:
   - Identify variables and their domains
   - Classify constraints (linear/nonlinear)
   - Understand objective function structure

2) Search Space Exploration:
   - Start with feasible solution
   - Identify improving directions
   - Consider boundary cases

3) Optimality Proof:
   - Local optimality conditions
   - Global optimality arguments
   - Constraint qualification

4) Solution Verification:
   - Check feasibility
   - Verify optimality conditions
   - Consider potential improvements

Provide:
1. Optimal value
2. Optimal assignment
3. Proof of optimality
4. Verification steps"""

    @staticmethod
    def model_counting_decomposition(formula: str, bound: int = None) -> str:
        return f"""Let's count models using systematic decomposition and verification.

FORMULA: {formula}
{f'Bound: up to {bound} models' if bound else 'Count all models'}

COUNTING PROCESS:

1) Formula Analysis:
   - Identify independent subformulas
   - Count variables and their domains
   - Recognize symmetries

2) Decomposition Strategy:
   - Break into subproblems
   - Handle each component
   - Combine results carefully

3) Counting Method:
   - For each subproblem:
     * Enumerate possibilities
     * Apply combinatorial reasoning
     * Consider symmetries
     * Track running total

4) Verification:
   - Check edge cases
   - Verify combinations
   - Confirm bounds
   - Double-check calculations

Provide:
1. Total model count
2. Breakdown of calculation
3. Verification steps"""