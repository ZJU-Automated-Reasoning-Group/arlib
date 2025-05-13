# SMTO: Satisfiability Modulo Theories and Oracles

## Overview

The `arlib/llm/smto` module provides an implementation of SMTO (Satisfiability Modulo Theories and Oracles) that leverages large language models as oracle handlers. This approach enables SMT solving for open programs with components lacking formal specifications, such as third-party libraries or deep learning models.

## Key Features

- **Two Oracle Modes**:
  - **Blackbox Mode**: Traditional SMTO where we only have input-output examples
  - **Whitebox Mode**: Enhanced SMTO that analyzes component information (docs, code)
- **Multi-provider support**: Compatible with OpenAI and Anthropic LLMs
- **Robust caching**: Persistent cache for oracle results to improve performance
- **Explanation capabilities**: Detailed logs of solving process for debugging
- **Multiple oracle types**: Support for LLM-based, function-based, and external oracles

## Usage Examples

### Blackbox Mode

Standard SMTO with input-output examples:

```python
import z3
from arlib.llm import OraxSolver, OracleInfo

# Initialize solver
solver = OraxSolver(
    provider="openai",  # or "anthropic"
    model="gpt-4",
    explanation_level="detailed"  # or "basic", "none"
)

# Register a blackbox oracle with examples
oracle = OracleInfo(
    name="strlen",
    input_types=[z3.StringSort()],
    output_type=z3.IntSort(),
    description="Calculate the length of a string",
    examples=[
        {"input": {"arg0": "hello"}, "output": "5"},
        {"input": {"arg0": "world!"}, "output": "6"}
    ]
)
solver.register_oracle(oracle)

# Add constraints
s = z3.String('s')
length = z3.Int('length')
strlen = z3.Function('strlen', z3.StringSort(), z3.IntSort())

solver.add_constraint(strlen(s) == length)
solver.add_constraint(length > 5)
solver.add_constraint(length < 10)

# Solve with oracle feedback
model = solver.check()
if model:
    print(f"Solution: s = {model[s]}, length = {model[length]}")
```

### Whitebox Mode

Enhanced SMTO that analyzes component source code, documentation, etc.:

```python
import z3
from arlib.llm import OraxSolver, WhiteboxOracleInfo, OracleAnalysisMode

# Initialize solver with whitebox analysis enabled
solver = OraxSolver(
    explanation_level="detailed",
    whitebox_analysis=True
)

# Source code for a password validator
source_code = """
def check_password(password):
    # Password must be at least 8 characters
    if len(password) < 8:
        return False
    
    # Password must contain a digit
    has_digit = False
    for char in password:
        if char.isdigit():
            has_digit = True
            break
    
    if not has_digit:
        return False
    
    # Password must contain a special character
    special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>/?"
    has_special = False
    for char in password:
        if char in special_chars:
            has_special = True
            break
            
    return has_special
"""

# Register a whitebox oracle with source code
password_oracle = WhiteboxOracleInfo(
    name="check_password",
    input_types=[z3.StringSort()],
    output_type=z3.BoolSort(),
    description="Check if a password meets security requirements",
    examples=[
        {"input": {"arg0": "password123"}, "output": "false"},
        {"input": {"arg0": "p@ssw0rd"}, "output": "true"}
    ],
    analysis_mode=OracleAnalysisMode.SOURCE_CODE,
    source_code=source_code
)
solver.register_oracle(password_oracle)

# Add constraints to find a valid password
password = z3.String('password')
check_pw = z3.Function('check_password', z3.StringSort(), z3.BoolSort())
solver.add_constraint(check_pw(password) == True)
solver.add_constraint(z3.Length(password) <= 12)

# Solve the constraint
model = solver.check()
if model:
    print(f"Valid password found: {model[password]}")
    print(f"Symbolic model: {solver.get_symbolic_model('check_password')}")
```

## Custom Function Oracles

You can use pure Python functions as oracles:

```python
from arlib.llm import OracleInfo, OracleType

def my_oracle_function(arg0):
    # Implement your logic here
    return len(arg0)

oracle = OracleInfo(
    name="my_oracle",
    input_types=[z3.StringSort()],
    output_type=z3.IntSort(),
    description="My custom oracle function",
    examples=[],
    oracle_type=OracleType.FUNCTION,
    function=my_oracle_function
)
```

## Whitebox Analysis Modes

The whitebox mode supports different analysis approaches:

```python
from arlib.llm import WhiteboxOracleInfo, OracleAnalysisMode

# Analyze component documentation
doc_oracle = WhiteboxOracleInfo(
    # Basic oracle info...
    analysis_mode=OracleAnalysisMode.DOCUMENTATION,
    documentation="Detailed API documentation..."
)

# Analyze source code
code_oracle = WhiteboxOracleInfo(
    # Basic oracle info...
    analysis_mode=OracleAnalysisMode.SOURCE_CODE,
    source_code="function source code..."
)

# Analyze binary (limited capability)
binary_oracle = WhiteboxOracleInfo(
    # Basic oracle info...
    analysis_mode=OracleAnalysisMode.BINARY,
    binary_code=binary_data
)

# Combined analysis (mixed mode)
mixed_oracle = WhiteboxOracleInfo(
    # Basic oracle info...
    analysis_mode=OracleAnalysisMode.MIXED,
    documentation="API docs...",
    source_code="Source code...",
    external_knowledge=["Known bug reports", "Forum discussions"]
)
```

## Advanced Features

### Persistent Caching

Enable persistent caching to improve performance:

```python
solver = OraxSolver(
    cache_dir="/path/to/cache"
)
```

### Explanation and Debugging

Get detailed explanations of the solving process:

```python
# After solving
explanations = solver.get_explanations()
for explanation in explanations:
    print(f"{explanation['message']}")
```

## Requirements

- Z3 SMT Solver
- One of the following LLM providers:
  - OpenAI (`pip install openai`)
  - Anthropic (`pip install anthropic`)

