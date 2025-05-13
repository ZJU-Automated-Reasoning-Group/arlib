"""Zhipu AI LLM implementation."""

from zhipuai import ZhipuAI
import datetime
import os
from typing import List, Optional, Union
from llm4opt.llm.base import EnvLoader, LLM


class ZhipuLLM(LLM):
    def __init__(self, model_name: str = "glm-4-flash", **kwargs):
        self.model_name = model_name
        self.api_key = EnvLoader.get_env("ZHIPU_API_KEY")
        self.client = ZhipuAI(api_key=self.api_key)
        self.tools = [{
            "type": "web_search",
            "web_search": {
                "enable": False,
            }
        }]
        self.kwargs = kwargs
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        if not self.api_key:
            raise ValueError("No API key found. Set ZHIPU_API_KEY environment variable or in .env file.")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            tools=self.tools,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError("Embedding functionality not implemented for ZhipuLLM yet")


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    msg = "Suppose we have: func(1) == 6, func(2) == 7, func(3) == 8, func(4) == 9. Write this function."
    start_time = datetime.datetime.now()
    llm = ZhipuLLM()
    result = llm.generate(msg)
    print(result)
    print(f"Time taken: {datetime.datetime.now() - start_time}")
