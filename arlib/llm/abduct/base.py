"""Base class for LLM implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import os
from dotenv import load_dotenv


class EnvLoader:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            load_dotenv()
            self._initialized = True
    
    @staticmethod
    def get_env(key: str, default: str = "") -> str:
        EnvLoader()  # Ensure environment is loaded
        return os.getenv(key, default)


class LLM(ABC):
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: Optional[int] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        pass
