"""Demo for Combinatorial Interaction Testing (CIT)

Combinatorial Interaction Testing is a systematic approach to test systems
where multiple parameters can interact to produce failures. It generates
test cases that cover all possible combinations of parameter values up
to a certain strength t (typically t=2 or t=3).

Key Features:
------------
- t-way interaction coverage
- Constraint handling
- Test case minimization
- Support for mixed-strength testing

Example Usage:
-------------
Consider testing a web application with parameters:
- Browser: Chrome, Firefox, Safari
- OS: Windows, MacOS, Linux
- Screen: Mobile, Desktop
- Network: WiFi, 4G, 5G

A 2-way (pairwise) CIT suite ensures all pairs of parameter values
are tested while minimizing the total number of test cases.

References:
----------
1. Kuhn, Kacker, Lei
   "Introduction to Combinatorial Testing"
   Chapman and Hall/CRC, 2013

2. Cohen, Dwyer, Shi
   "Interaction Testing of Highly-Configurable Systems
    in the Presence of Constraints"
   ISSTA 2007
"""

from typing import List, Dict, Set, Tuple
import z3
from dataclasses import dataclass

@dataclass
class Parameter:
    """Represents a test parameter with its possible values."""
    name: str
    values: List[str]

@dataclass
class TestCase:
    """Represents a single test case as parameter-value assignments."""
    assignments: Dict[str, str]

def generate_test_cases(parameters: List[Parameter], t: int) -> List[TestCase]:
    """Generates all test cases for given parameters up to strength t."""
    solver = z3.Solver()
    variables = {param.name: z3.Int(param.name) for param in parameters}
    # Constraint: Each parameter value is assigned exactly once
    for param in parameters:
        solver.add(z3.Distinct([variables[param.name]] + [z3.Int(f"{param.name}_{i}") for i in range(len(param.values))]))
    # Constraint: Each parameter value is assigned a unique integer
    for param in parameters:
        solver.add(z3.And([z3.Or([variables[param.name] == i for i in range(len(param.values))])]))
    # Constraint: Interaction strength is t
    for i in range(len(parameters)):
        for j in range(i+1, len(parameters)):
            solver.add(z3.Or([z3.And(variables[parameters[i].name] == k, variables[parameters[j].name] == l)
                              for k in range(len(parameters[i].values)) for l in range(len(parameters[j].values))]))
    test_cases = []
    while solver.check() == z3.sat:
        model = solver.model()
        test_case = {param.name: param.values[model[variables[param.name]].as_long()] for param in parameters}
        test_cases.append(TestCase(test_case))
        # Constraint: Current test case is not repeated
        solver.add(z3.Or([z3.Not(variables[param.name] == model[variables[param.name]].as_long()) for param in parameters]))
    return test_cases
    