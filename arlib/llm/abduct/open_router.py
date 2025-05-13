"""OpenRouter implementation."""

import os
from typing import List, Optional, Union

from openai import OpenAI
from llm4opt.llm.base import LLM, EnvLoader


class OpenRouter(LLM):
    def __init__(self, 
                model_name: str = "deepseek/deepseek-chat-v3-0324",
                api_key: Optional[str] = None,
                site_url: Optional[str] = None,
                site_name: Optional[str] = None,
                **kwargs):
        self.model_name = model_name
        
        # Set API key from parameter or environment variable
        self.api_key = api_key or EnvLoader.get_env("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required via api_key parameter or OPENROUTER_API_KEY environment variable")
        
        # Set site details for rankings
        self.site_url = site_url or "https://llm4opt.ai"
        self.site_name = site_name or "llm4opt"
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.kwargs = kwargs
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        # Prepare extra headers for OpenRouter
        extra_headers = {
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        
        # Prepare parameters for the API call
        params = {
            "extra_headers": extra_headers,
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop is not None:
            params["stop"] = stop
            
        # Add any additional kwargs
        params.update(kwargs)
        
        # Make the API call
        response = self.client.chat.completions.create(**params)
        
        # Extract and return the generated text
        return response.choices[0].message.content.strip()
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError("Embedding functionality not implemented for OpenRouter yet")
