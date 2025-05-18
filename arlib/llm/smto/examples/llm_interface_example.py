"""
Example usage of the LLM interface with different providers
"""

import logging
import os
from arlib.llm.smto.llm_providers import LLMConfig, LLMInterface
from arlib.llm.smto.llm_factory import create_llm

# Configure logging
logging.basicConfig(level=logging.INFO)

def openai_example():
    """Example using OpenAI"""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.1,
        max_tokens=500
    )
    
    llm = create_llm(config)
    
    # Single prompt example
    prompt = "Explain the concept of SMT solving in one sentence."
    result = llm.complete(prompt)
    print(f"OpenAI single prompt result:\n{result}\n")
    
    # Chat example
    messages = [
        {"role": "system", "content": "You are a helpful assistant that explains technical concepts."},
        {"role": "user", "content": "What is SMT solving?"},
        {"role": "assistant", "content": "SMT solving is a technique to determine if a logical formula is satisfiable."},
        {"role": "user", "content": "Can you provide a simple example?"}
    ]
    
    result = llm.chat_complete(messages)
    print(f"OpenAI chat result:\n{result}\n")

def anthropic_example():
    """Example using Anthropic"""
    config = LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.1
    )
    
    llm = create_llm(config)
    
    # Single prompt example with system prompt
    prompt = "Explain the concept of SMT solving in one sentence."
    system_prompt = "You are a helpful assistant that explains technical concepts concisely."
    result = llm.complete(prompt, system_prompt=system_prompt)
    print(f"Anthropic single prompt result:\n{result}\n")

def gemini_example():
    """Example using Google Gemini"""
    config = LLMConfig(
        provider="gemini",
        model="gemini-1.5-pro",
        temperature=0.2
    )
    
    llm = create_llm(config)
    
    # Single prompt example
    prompt = "Explain the concept of SMT solving in one sentence."
    result = llm.complete(prompt)
    print(f"Gemini single prompt result:\n{result}\n")

def openrouter_example():
    """Example using OpenRouter"""
    config = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3-opus-20240229",  # Use any model available on OpenRouter
        temperature=0.1
    )
    
    llm = create_llm(config)
    
    # Single prompt example
    prompt = "Explain the concept of SMT solving in one sentence."
    result = llm.complete(prompt)
    print(f"OpenRouter single prompt result:\n{result}\n")

if __name__ == "__main__":
    # Uncomment the example you want to run
    # Make sure you have the appropriate API keys set in your environment
    
    # openai_example()
    # anthropic_example()
    # gemini_example()
    # openrouter_example()
    
    print("Select which example to run by uncommenting the appropriate line in the script.") 