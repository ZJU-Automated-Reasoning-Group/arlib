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
import openai
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM API"""
    api_key: str
    model: str = "gpt-4"  # Default to GPT-4
    temperature: float = 0.1
    max_tokens: int = 1000

class LogicLLM:
    def __init__(self, config: LLMConfig):
        """Initialize LogicLLM with configuration"""
        self.config = config
        openai.api_key = config.api_key
        
    async def query_llm(self, prompt: str) -> str:
        """Query LLM with given prompt"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
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
            import re
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
    import os
    
    config = LLMConfig(api_key=os.getenv("OPENAI_API_KEY"))
    llm = LogicLLM(config)
    
    async def run_demo():
        # Example formula
        formula = "x > 0 & x < 10 & x + y = 5"
        
        # Check satisfiability
        is_sat = await llm.check_sat(formula)
        print(f"Is satisfiable: {is_sat}")
        
        # Simplify formula
        simplified = await llm.simplify_formula(formula)
        print(f"Simplified: {simplified}")
        
    asyncio.run(run_demo())

if __name__ == "__main__":
    demo()
    