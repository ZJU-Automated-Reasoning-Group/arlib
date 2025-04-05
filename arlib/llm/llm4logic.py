"""Query LLM for Solving Logical Reasoning Problems
Tasks:
- Satisfiability checking
- Abductive reasoning
- Consequence finding
- Quantifier elimination
- Craig interpolant generation
- Formula simplifications
- Model counting
- Optimization Modulo Theories
- ...
"""

import logging
from typing import Optional, List, Dict, Any, Union
import re
import os
import asyncio
from dataclasses import dataclass

from arlib.llm.llm_factory import LLMConfig, create_llm, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicLLM:
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LogicLLM with configuration"""
        self.config = config or LLMConfig()
        self.llm = create_llm(self.config)

    async def query_llm(self, prompt: str) -> str:
        """Query LLM with given prompt"""
        try:
            messages = [{"role": "user", "content": prompt}]
            return await asyncio.to_thread(self.llm.chat, messages)
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise

    async def check_sat(self, formula: str, logic: str = "QF_LIA") -> bool:
        """Check satisfiability of a formula"""
        prompt = f"""Given the following formula in {logic}:
{formula}
Is this formula satisfiable? Explain your reasoning step by step."""

        response = await self.query_llm(prompt)
        return "satisfiable" in response.lower()

    async def find_abduction(self, background: str, observation: str,
                             variables: List[str]) -> str:
        """Find abductive explanation"""
        prompt = f"""Given:
Background knowledge: {background}
Observation: {observation}
Variables to use: {', '.join(variables)}

Find the simplest explanation (abductive hypothesis) that, together with the 
background knowledge, logically entails the observation."""

        return await self.query_llm(prompt)

    async def eliminate_quantifiers(self, formula: str) -> str:
        """Perform quantifier elimination"""
        prompt = f"""Eliminate quantifiers from the following formula:
{formula}

Show the step-by-step process and provide the final quantifier-free formula."""

        return await self.query_llm(prompt)

    async def compute_interpolant(self, formula_a: str, formula_b: str) -> str:
        """Compute Craig interpolant"""
        prompt = f"""Find a Craig interpolant for the following formulas:
Formula A: {formula_a}
Formula B: {formula_b}

The interpolant should:
1. Use only variables common to both formulas
2. Be implied by formula A
3. Be inconsistent with formula B"""

        return await self.query_llm(prompt)

    async def simplify_formula(self, formula: str, logic: str = "QF_LIA") -> str:
        """Simplify logical formula"""
        prompt = f"""Simplify the following formula in {logic}:
{formula}

Provide the simplified formula and explain the simplification steps."""

        return await self.query_llm(prompt)

    async def count_models(self, formula: str, bound: Optional[int] = None) -> int:
        """Count models of a formula"""
        prompt = f"""Count the number of models for the formula:
{formula}
{"(Up to " + str(bound) + " models)" if bound else ""}

Show your counting process."""

        response = await self.query_llm(prompt)
        try:
            # Extract the number from response
            numbers = re.findall(r'\d+', response)
            return int(numbers[0]) if numbers else 0
        except:
            return 0

    async def optimize(self, formula: str, objective: str,
                       minimize: bool = True) -> Dict[str, Any]:
        """Solve optimization problem"""
        prompt = f"""Solve the optimization problem:
Formula (constraints): {formula}
Objective function: {objective}
Goal: {"Minimize" if minimize else "Maximize"}

Provide:
1. Optimal value
2. Assignment to variables
3. Proof of optimality"""

        response = await self.query_llm(prompt)
        return {"response": response}  # Parse response as needed


def demo():
    """Demo usage of LogicLLM"""
    import asyncio

    # Create config with provider specified from environment or default to OpenAI
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Select appropriate model based on provider
    model_map = {
        "openai": "gpt-4",
        "anthropic": "claude-3-opus-20240229",
        "gemini": "gemini-pro",
        "zhipu": "glm-4"
    }

    model = model_map.get(provider, "gpt-4")

    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=0.1
    )

    llm = LogicLLM(config)

    async def run_demo():
        # Example formula
        formula = "x > 0 & x < 10 & x + y = 5"

        print(f"Using LLM provider: {provider} with model: {model}")

        # Check satisfiability
        is_sat = await llm.check_sat(formula)
        print(f"Is satisfiable: {is_sat}")

        # Simplify formula
        simplified = await llm.simplify_formula(formula)
        print(f"Simplified: {simplified}")

    asyncio.run(run_demo())


if __name__ == "__main__":
    demo()
