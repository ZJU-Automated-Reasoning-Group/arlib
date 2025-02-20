"""Query LLM for Solving Logical Reasoning Problems
- Satisfiability of a logical formula
- Abductive reasoning
- Consequence finding
"""

import openai

def query_llm(prompt: str) -> str:
    """Query LLM for a given prompt"""
    return openai.Completion.create(prompt=prompt)


def check_sat(formula: str) -> bool:
    """Check satisfiability of a logical formula"""
    prompt = f"Check if the following logical formula is satisfiable: {formula}"
    response = query_llm(prompt)
    return response.choices[0].text.strip() == "True"

def check_implication(fmla, fmlb) -> bool:
    """Check if fmla implies fmlb"""
    prompt = f"Check if the first formula subsumes the second one: {fmla} -> {fmlb}"
    response = query_llm(prompt)
    return response.choices[0].text.strip() == "True"

def check_equivalence(fmla, fmlb) -> bool:
    """Check if fmla and fmlb are equivalent"""
    prompt = f"Check if the following two formulas are equivalent: {fmla} <-> {fmlb}"
    response = query_llm(prompt)
    return response.choices[0].text.strip() == "True"
