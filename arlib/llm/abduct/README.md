# LLM-based Abduction with SMT Validation

This module provides a framework for evaluating Large Language Models (LLMs) on abductive inference tasks with SMT constraints validation, using ZhipuAI as the LLM provider and Z3 for validation.

## Overview

The framework implements abductive reasoning using LLMs, where:

1. The **premise** and **conclusion** are given as SMT constraints
2. The LLM is tasked with generating a hypothesis that explains how the conclusion follows from the premise
3. The generated hypothesis is validated using Z3 to check for correctness

Abductive reasoning follows this pattern:
- Given: Premises (Γ) and a desired conclusion (φ)
- Find: A hypothesis (ψ) such that:
  - Γ ∧ ψ is satisfiable (consistency)
  - Γ ∧ ψ |= φ (sufficiency to imply the conclusion)

## Components

The framework consists of the following key components:

- **LLMAbductor**: Uses LLMs to generate abductive hypotheses for given SMT constraints
- **AbductionProblem**: Represents an abduction problem with SMT constraints
- **AbductionEvaluator**: Evaluates LLM performance on abduction tasks

## Installation

Install the required dependencies:

```bash
pip install z3-solver numpy python-dotenv zhipuai
```

## Setup

Configuration is managed via environment variables. Create a `.env` file in your project directory:

```
ZHIPU_API_KEY=your_zhipu_api_key
```

## Usage

### Run Demo

The easiest way to get started is to run the demo script, which uses ZhipuAI to solve example abduction problems:

```bash
python demo.py
```

### API Usage Example

```python
from arlib.llm.abduct import AbductionProblem, LLMAbductor
from arlib.llm.abduct.zhipu import ZhipuLLM
import z3

# Create a problem
x, y = z3.Ints('x y')
premise = z3.And(x >= 0, y >= 0)
conclusion = x + y <= 10
problem = AbductionProblem(
    premise=premise,
    conclusion=conclusion,
    description="Find conditions that ensure the sum is at most 10",
    variables=[x, y]
)

# Initialize LLM and abductor
llm = ZhipuLLM(model_name="glm-4-flash")
abductor = LLMAbductor(llm=llm)

# Generate abduction
result = abductor.abduce(problem)

# Check results
print(f"Hypothesis: {result.hypothesis}")
print(f"Consistent: {result.is_consistent}")
print(f"Sufficient: {result.is_sufficient}")
print(f"Valid: {result.is_valid}")
```

### Evaluation

You can evaluate the LLM on a list of problems and save the results:

```python
from arlib.llm.abduct import AbductionEvaluator

# problems = [...]  # List of AbductionProblem
# abductor = ...    # An instance of LLMAbductor

evaluator = AbductionEvaluator(abductor, problems)
metrics = evaluator.evaluate()
print(metrics)
evaluator.save_results("results.json")
```

## LLM Support

Currently supported LLM provider:

- **Zhipu AI**: Chinese LLM provider with models like GLM-4. Only ZhipuAI is supported out of the box.

## Extending the Framework

### Adding New LLM Providers

To add a new provider, subclass the `LLM` base class in `base.py`:

```python
from arlib.llm.abduct.base import LLM

class MyCustomLLM(LLM):
    def __init__(self, model_name, **kwargs):
        # Initialize your LLM client
        pass
        
    def generate(self, prompt, temperature=0.7, max_tokens=None, stop=None, **kwargs):
        # Implement generation logic
        pass
        
    def get_embedding(self, text, **kwargs):
        # Implement embedding logic
        pass
```

## Notes

- Only ZhipuAI is supported in the current implementation.
- Example problems and usage are provided in `demo.py`.
- There are **no built-in benchmark generation functions** in this module.
- For custom benchmarks, define your own list of `AbductionProblem` instances. 