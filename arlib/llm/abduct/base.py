"""Abduction LLM adapter that wraps `arlib.llm.llmtool` for querying providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import os
from dotenv import load_dotenv

from arlib.llm.llmtool.LLM_tool import LLMTool, LLMToolInput, LLMToolOutput
from arlib.llm.llmtool.logger import Logger


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
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> str:
        ...

    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        ...


class _AbductionLLMTool(LLMTool):
    """A minimal LLMTool wrapper that takes a prompt and returns raw text."""

    class _Input(LLMToolInput):
        def __init__(self, prompt: str):
            self.prompt = prompt

        def __hash__(self):
            return hash(self.prompt)

    class _Output(LLMToolOutput):
        def __init__(self, text: str):
            self.text = text

    def _get_prompt(self, input: "_AbductionLLMTool._Input") -> str:
        return input.prompt

    def _parse_response(
        self, response: str, input: "_AbductionLLMTool._Input" = None
    ) -> "_AbductionLLMTool._Output":
        # Return raw text as output
        return _AbductionLLMTool._Output(response)


class LLMViaTool(LLM):
    """Concrete LLM that reuses `arlib.llm.llmtool` for querying providers."""

    def __init__(self, model_name: str, temperature: float = 0.7, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        log_dir = os.environ.get("ARLIB_LOG_DIR", ".arlib_logs")
        logger = Logger(os.path.join(log_dir, "abduction_llm.log"))
        # The LLMTool keeps provider selection in its LLM_utils based on model name
        self.tool = _AbductionLLMTool(
            model_name=model_name,
            temperature=temperature,
            language="en",
            max_query_num=3,
            logger=logger,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> str:
        # LLMTool ignores max_tokens/stop; we pass the prompt through
        input_obj = _AbductionLLMTool._Input(prompt)
        output_obj = self.tool.invoke(input_obj)
        return output_obj.text if output_obj else ""

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        raise NotImplementedError("Embedding via llmtool not provided here")
