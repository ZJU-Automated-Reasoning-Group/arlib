"""
A few prompts related to SMT
"""

"""
Provide a step-by-step solution for the following SMT-LIB2 problem, 
detailing each logical step and decision made.
"""

# Can Language Models Pretend Solvers? Logic Code Simulation with LLMs, SETTA 24.
smt_dual_chains_of_logic = """
Given the following code in Z3 solver, you are asked to determine the
output of the code with #sat or #unsat. Follow instructions below:

1. List and understand variables and constraints added to the solver.

2. Based on the constraints, if I say the result is 'sat', is it
correct? Can you find satisfied assignments for variables? Let's
think step by step.

3. Based on the constraints, if I say the result is 'unsat', is it
correct? Can you find conflict constraints? Let's think step by step.
4. Which is more correct and reasonable based on the above aspects?
Answer your preferred hypothesis with #sat or #unsat. Let's think
step by step.

Code：[[CODE]]
"""
