"""
Simplified LLM adapter that provides compatibility with the removed llm_factory and llm_providers
by using the LLM class from llmtool.LLM_utils as the backend.
"""

import logging
import os
import tempfile
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Import from llmtool
from arlib.llm.llmtool.logger import Logger
from arlib.llm.llmtool.LLM_utils import LLM


@dataclass
class LLMConfig:
    """Configuration for LLM API - simplified version"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000
    provider: str = "openai"
    base_url: Optional[str] = None
    system_role: str = "You are a experienced programmer and good at understanding programs written in mainstream programming languages."

    def __post_init__(self):
        # Set up API keys based on model/provider
        if "gpt" in self.model or "o3-mini" in self.model:
            if self.api_key is None:
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if self.api_key is None:
                    logging.warning("No API key provided and OPENAI_API_KEY not found in environment variables")
        elif "deepseek" in self.model:
            if self.api_key is None:
                self.api_key = os.environ.get("DEEPSEEK_API_KEY2")
                if self.api_key is None:
                    logging.warning("No API key provided and DEEPSEEK_API_KEY2 not found in environment variables")
        elif "gemini" in self.model:
            if self.api_key is None:
                self.api_key = os.environ.get("GOOGLE_API_KEY")
                if self.api_key is None:
                    logging.warning("No API key provided and GOOGLE_API_KEY not found in environment variables")
        elif "claude" in self.model:
            # Claude uses AWS credentials, no specific API key needed here
            pass


class LLMInterface:
    """Simplified LLM interface using LLM_utils.LLM as backend"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM provider with configuration"""
        self.config = config or LLMConfig()
        
        # Create a logger
        log_file = os.path.join(tempfile.gettempdir(), "llm_adapter.log")
        self.logger = Logger(log_file)
        
        # Initialize LLM from LLM_utils
        self.llm = LLM(
            online_model_name=self.config.model,
            logger=self.logger,
            temperature=self.config.temperature,
            system_role=self.config.system_role
        )

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate completion from a prompt with optional system instructions"""
        # If system_prompt is provided, temporarily update the system role
        original_system_role = self.llm.systemRole
        if system_prompt:
            self.llm.systemRole = system_prompt
        
        try:
            output, _, _ = self.llm.infer(prompt, is_measure_cost=False)
            return output
        finally:
            # Restore original system role
            self.llm.systemRole = original_system_role
    
    def chat_complete(self, messages: List[Dict[str, str]]) -> str:
        """Generate completion from a chat history"""
        # Extract system message if present
        system_prompt = None
        user_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                user_messages.append(message.get("content", ""))
        
        # Combine user messages into a single prompt
        combined_prompt = "\n".join(user_messages)
        
        return self.complete(combined_prompt, system_prompt)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Alternative method name for compatibility - same as complete()"""
        return self.complete(prompt, system_prompt)

    def infer_with_cost(self, message: str) -> tuple[str, int, int]:
        """Direct access to LLM.infer method with cost measurement"""
        return self.llm.infer(message, is_measure_cost=True)


def create_llm(config: Optional[LLMConfig] = None) -> LLMInterface:
    """Factory function to create LLM interface based on config"""
    if config is None:
        config = LLMConfig()
    
    return LLMInterface(config) 