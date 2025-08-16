import json
import os
import time
import concurrent.futures
from typing import Tuple
from arlib.llm.llmtool.logger import Logger

# Optional SDKs
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import tiktoken
except ImportError:
    tiktoken = None
try:
    from botocore.config import Config
    import boto3
except ImportError:
    Config = boto3 = None
try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None


class LLM:
    """Multi-provider LLM inference: Gemini, OpenAI, DeepSeek, Claude, GLM"""

    def __init__(self, online_model_name: str, logger: Logger, temperature: float = 0.0,
                 system_role="You are an experienced programmer.") -> None:
        self.online_model_name = online_model_name
        self.temperature = temperature
        self.systemRole = system_role
        self.logger = logger

        # Token encoding for cost measurement
        if tiktoken:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
        else:
            self.encoding = type('_DummyEncoding', (), {'encode': lambda _, s: s.encode("utf-8")})()

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        self.logger.print_log(self.online_model_name, "is running")
        output = ""
        if "gemini" in self.online_model_name:
            output = self.infer_with_gemini(message)
        elif "gpt" in self.online_model_name:
            output = self.infer_with_openai_model(message)
        elif "o3-mini" in self.online_model_name:
            output = self.infer_with_o3_mini_model(message)
        elif "glm" in self.online_model_name or "zhipu" in self.online_model_name:
            # Zhipu/GLM models (e.g., glm-4, glm-4-flash)
            output = self.infer_with_glm_model(message)
        elif "claude" in self.online_model_name:
            output = self.infer_with_claude(message)
        elif "deepseek" in self.online_model_name:
            output = self.infer_with_deepseek_model(message)
        else:
            raise ValueError("Unsupported model name")

        input_token_cost = (
            0
            if not is_measure_cost
            else len(self.encoding.encode(self.systemRole))
            + len(self.encoding.encode(message))
        )
        output_token_cost = (
            0 if not is_measure_cost else len(self.encoding.encode(output))
        )
        return output, input_token_cost, output_token_cost

    def _retry_api_call(self, call_func, timeout=100, max_retries=5):
        """Common retry logic for all API calls"""
        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_func)
                    result = future.result(timeout=timeout)
                    if result:
                        return result
            except concurrent.futures.TimeoutError:
                self.logger.print_log("Operation timed out")
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)
        return ""

    def infer_with_gemini(self, message: str) -> str:
        """Infer using Gemini model"""
        if not genai:
            self.logger.print_log("Gemini SDK not installed")
            return ""

        model = genai.GenerativeModel("gemini-pro")
        def call_api():
            return model.generate_content(
                f"{self.systemRole}\n{message}",
                safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}],
                generation_config=genai.types.GenerationConfig(temperature=self.temperature)
            ).text

        return self._retry_api_call(call_api, timeout=50)

    def infer_with_openai_model(self, message):
        """Infer using OpenAI model"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").split(":")[0])
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api)

    def infer_with_o3_mini_model(self, message):
        """Infer using o3-mini model"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").split(":")[0])
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}]
            ).choices[0].message.content

        return self._retry_api_call(call_api)

    def infer_with_deepseek_model(self, message):
        """Infer using DeepSeek model"""
        def call_api():
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY2"),
                          base_url="https://api.deepseek.com")
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api, timeout=300)

    def infer_with_claude(self, message):
        """Infer using Claude model via AWS Bedrock"""
        if not boto3 or not Config:
            self.logger.print_log("boto3/botocore not installed for Claude")
            return ""

        model_id = ("anthropic.claude-3-5-sonnet-20241022-v2:0" if "3.5" in self.online_model_name
                   else "us.anthropic.claude-3-7-sonnet-20250219-v1:0")

        def call_api():
            client = boto3.client("bedrock-runtime", region_name="us-west-2",
                                config=Config(read_timeout=100))
            body = json.dumps({
                "messages": [{"role": "assistant", "content": self.systemRole},
                           {"role": "user", "content": message}],
                "max_tokens": 4000,
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": self.temperature,
                "top_k": 50
            })
            response = client.invoke_model(modelId=model_id, contentType="application/json", body=body)
            return json.loads(response["body"].read().decode("utf-8"))["content"][0]["text"]

        return self._retry_api_call(call_api)


    def infer_with_glm_model(self, message):
        """Infer using GLM model"""
        if not ZhipuAI:
            self.logger.print_log("ZhipuAI SDK not installed")
            return ""

        def call_api():
            client = ZhipuAI(api_key=os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY"))
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api)
