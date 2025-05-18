"""Zhipu AI LLM implementation."""

import datetime
# import os
import sys
#import json
# import re
from typing import List, Optional, Union, Dict, Any
from arlib.llm.abduct.base import EnvLoader, LLM

class ZhipuLLM(LLM):
    def __init__(self, model_name: str = "glm-4-flash", **kwargs):
        self.model_name = model_name
        
        # Get API key using EnvLoader (which handles loading from .env)
        self.api_key = EnvLoader.get_env("ZHIPU_API_KEY")
        
        # Import ZhipuAI here to avoid import errors if the library is not installed
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
        except ImportError:
            print("Error: zhipuai library not installed. Please run 'pip install zhipuai'")
            sys.exit(1)
        except Exception as e:
            print(f"Error initializing ZhipuAI client: {e}")
            sys.exit(1)
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        """
        Generate text using Zhipu AI models.
        
        Args:
            prompt: The input prompt for generation
            temperature: The temperature for sampling (0-1)
            max_tokens: Maximum number of tokens to generate
            stop: Stop sequences
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            str: The generated text
        """
        if not self.api_key:
            raise ValueError("No API key found. Set ZHIPU_API_KEY environment variable or in .env file.")
        
        # Simplify the prompt if it's too long
        processed_prompt = prompt
        if len(prompt) > 4000:
            processed_prompt = prompt[:4000]
        
        try:
            # Create very minimal API call with only required parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": processed_prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            error_message = f"Zhipu API Error: {error_str}"
            print(error_message)
            return f"Error: {error_str}"
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError("Embedding functionality not implemented for ZhipuLLM yet")


# Example usage
if __name__ == "__main__":
    # Test the Zhipu LLM implementation
    #import sys
    #import os
    #sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    msg = "Suppose we have: func(1) == 6, func(2) == 7, func(3) == 8, func(4) == 9. Write this function."
    start_time = datetime.datetime.now()
    llm = ZhipuLLM()
    result = llm.generate(msg)
    print(result)
    print(f"Time taken: {datetime.datetime.now() - start_time}")
