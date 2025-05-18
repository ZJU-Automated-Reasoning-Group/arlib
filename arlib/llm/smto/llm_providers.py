"""
LLM provider interface and implementations for SMTO.
"""

import logging
import os
from typing import Optional, List, Dict, Any, Union
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

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import requests
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM API"""
    api_key: Optional[str] = None
    model: str = "gpt-4"  # Default model
    temperature: float = 0.1
    max_tokens: int = 1000
    provider: str = "openai"  # Default provider
    base_url: Optional[str] = None  # For OpenRouter or custom endpoints

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
        elif self.provider == "gemini":
            return os.environ.get("GOOGLE_API_KEY")
        elif self.provider == "openrouter":
            return os.environ.get("OPENROUTER_API_KEY")
        return None
    
    def _init_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
            kwargs = {}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            return OpenAI(api_key=self.api_key, **kwargs)
        
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
            return anthropic.Anthropic(api_key=self.api_key)
        
        elif self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
            genai.configure(api_key=self.api_key)
            return genai
        
        elif self.provider == "openrouter":
            if not OPENROUTER_AVAILABLE:
                raise ImportError("Requests package required for OpenRouter. Install with 'pip install requests'")
            return None  # We'll use direct API calls for OpenRouter
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate completion from a prompt with optional system instructions"""
        return self._process_request(prompt=prompt, system_prompt=system_prompt)
    
    def chat_complete(self, messages: List[Dict[str, str]]) -> str:
        """Generate completion from a chat history"""
        return self._process_request(messages=messages)
    
    def _process_request(self, 
                         prompt: Optional[str] = None, 
                         messages: Optional[List[Dict[str, str]]] = None,
                         system_prompt: Optional[str] = None) -> str:
        """Process LLM request based on provider and input type"""
        try:
            # Handle completion or chat based on provided arguments
            if prompt is not None:
                # Convert single prompt to messages format if needed
                if messages is None:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
            
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                # Convert standard format to Anthropic format if needed
                anthropic_messages = []
                for msg in messages:
                    role = msg["role"]
                    # Anthropic only supports user/assistant roles
                    if role == "system":
                        # Prepend system message to first user message
                        for i, m in enumerate(messages):
                            if m["role"] == "user":
                                anthropic_messages.append({
                                    "role": "user", 
                                    "content": f"{msg['content']}\n\n{m['content']}"
                                })
                                # Skip this user message when we process the list
                                messages[i]["processed"] = True
                                break
                    elif role == "user" and not msg.get("processed"):
                        anthropic_messages.append({"role": "user", "content": msg["content"]})
                    elif role == "assistant":
                        anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                
                response = self.client.messages.create(
                    model=self.config.model,
                    messages=anthropic_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.content[0].text
            
            elif self.provider == "gemini":
                if len(messages) == 1 and messages[0]["role"] == "user":
                    # Use completion API for single user message
                    model = self.client.GenerativeModel(self.config.model)
                    response = model.generate_content(messages[0]["content"], 
                                                     generation_config={
                                                         "temperature": self.config.temperature,
                                                         "max_output_tokens": self.config.max_tokens
                                                     })
                    return response.text
                else:
                    # Use chat API for multi-turn conversations
                    model = self.client.GenerativeModel(self.config.model)
                    chat = model.start_chat()
                    
                    # Process messages in order
                    gemini_messages = []
                    for msg in messages:
                        if msg["role"] == "user":
                            gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                        elif msg["role"] == "assistant":
                            gemini_messages.append({"role": "model", "parts": [msg["content"]]})
                    
                    # Add conversation history to the chat
                    for msg in gemini_messages[:-1]:
                        if msg["role"] == "user":
                            chat.send_message(msg["parts"][0])
                        # Model messages are added automatically after user messages
                    
                    # Send the final message and get response
                    response = chat.send_message(
                        gemini_messages[-1]["parts"][0],
                        generation_config={
                            "temperature": self.config.temperature,
                            "max_output_tokens": self.config.max_tokens
                        }
                    )
                    return response.text
            
            elif self.provider == "openrouter":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                base_url = self.config.base_url or "https://openrouter.ai/api/v1"
                
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"OpenRouter API error: {response.status_code} {response.text}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logging.error(f"Error processing request with {self.provider}: {str(e)}")
            raise 