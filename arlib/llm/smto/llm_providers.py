"""
LLM provider interface and implementations for SMTO.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Import LLM libraries conditionally
try:
    from openai import OpenAI
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


class LLMInterface:
    """Common interface for LLM interactions"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM provider with configuration"""
        self.config = config or LLMConfig()
        self.provider = self.config.provider.lower()
        self.api_key = self.config.api_key or self._get_api_key()
        self.client = self._init_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables"""
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        return None
    
    def _init_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
            return anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt with specified system prompt"""
        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                messages = []
                # Anthropic doesn't have a dedicated system message,
                # so we prepend it to the user message if provided
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{prompt}"
                
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.messages.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.content[0].text
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logging.error(f"Error generating text with {self.provider}: {str(e)}")
            raise 