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