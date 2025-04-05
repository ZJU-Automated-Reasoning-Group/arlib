import logging
import os
from typing import Optional, List, Dict, Any, Union
import openai
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import other LLM libraries
import anthropic

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import zhipuai
except ImportError:
    zhipuai = None


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
        self.config = config
        openai.api_key = config.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        response = openai.Completion.create(prompt=prompt, **params)
        return response.choices[0].text.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        response = openai.ChatCompletion.create(messages=messages, **params)
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        response = self.client.completions.create(prompt=prompt, **params)
        return response.completion.strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": msg["content"]})

        params = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        response = self.client.messages.create(messages=anthropic_messages, **params)
        return response.content[0].text


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    def __init__(self, config: LLMConfig):
        if genai is None:
            raise ImportError(
                "Google Generative AI package not installed. Install with 'pip install google-generativeai'")
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(model_name=config.model)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.generate_content(prompt,
                                               temperature=self.config.temperature,
                                               max_output_tokens=self.config.max_tokens,
                                               **kwargs)
        return response.text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        chat = self.model.start_chat(history=gemini_messages)
        response = chat.send_message("",
                                     temperature=self.config.temperature,
                                     max_output_tokens=self.config.max_tokens,
                                     **kwargs)
        return response.text


class ZhipuProvider(LLMProvider):
    """Zhipu AI API provider"""

    def __init__(self, config: LLMConfig):
        if zhipuai is None:
            raise ImportError("Zhipu AI package not installed. Install with 'pip install zhipuai'")
        self.config = config
        zhipuai.api_key = config.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        response = zhipuai.model.invoke(
            model=self.config.model,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        return response.get("response", "")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert to Zhipu format if needed
        response = zhipuai.model.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")


def create_llm(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Factory function to create LLM provider based on config"""
    if config is None:
        config = LLMConfig()

    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "zhipu": ZhipuProvider,
    }

    provider_class = providers.get(config.provider.lower())
    if provider_class is None:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    return provider_class(config)
