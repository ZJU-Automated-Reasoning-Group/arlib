"""
LLM provider factory for SMTO (Satisfiability Modulo Theories and Oracles)
This module provides a unified interface to various LLM providers for oracle handling.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import LLM libraries conditionally
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM API"""
    api_key: Optional[str] = None
    model: str = "gpt-4"  # Default model
    temperature: float = 0.1
    max_tokens: int = 1000
    provider: str = "openai"  # Default provider

    def __post_init__(self):
        # If no API key provided, try to get from environment variables
        if self.api_key is None:
            env_var_name = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.environ.get(env_var_name)
            if self.api_key is None:
                logging.warning(f"No API key provided and {env_var_name} not found in environment variables")


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion with message history"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
        self.config = config
        self.client = openai.OpenAI(api_key=config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        # Modern API requires messages
        response = self.client.completions.create(prompt=prompt, **params)
        return response.choices[0].text.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        response = self.client.chat.completions.create(messages=messages, **params)
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""

    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        # Anthropic doesn't support pure completions, convert to messages format
        response = self.client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        return response.content[0].text.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert standard format to Anthropic format if needed
        anthropic_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": msg["content"]})

        response = self.client.messages.create(
            model=self.config.model,
            messages=anthropic_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        return response.content[0].text.strip()


def create_llm(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Factory function to create LLM provider based on config"""
    if config is None:
        config = LLMConfig()

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    provider_class = providers.get(config.provider.lower())
    if provider_class is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    return provider_class(config)
