# Imports
from pathlib import Path
from typing import Tuple
# Optional provider SDKs (guarded)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None
try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None
# import signal
# import sys
try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None
import time
import os
import concurrent.futures
from functools import partial
# import threading

import json
try:
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
    import boto3
except Exception:  # pragma: no cover
    Config = None
    BotoCoreError = ClientError = None
    boto3 = None
from arlib.llm.llmtool.logger import Logger

try:
    from zhipuai import ZhipuAI
except Exception:  # pragma: no cover
    ZhipuAI = None


class LLM:
    """
    An online inference model using different LLMs:
    - Gemini
    - OpenAI: GPT-3.5, GPT-4, o3-mini
    - DeepSeek: V3, R1
    - Claude: 3.5 and 3.7
    """

    def __init__(
        self,
        online_model_name: str,
        logger: Logger,
        temperature: float = 0.0,
        system_role="You are a experienced programmer and good at understanding programs written in mainstream programming languages.",
    ) -> None:
        self.online_model_name = online_model_name
        if tiktoken is not None:
            self.encoding = tiktoken.encoding_for_model(
                "gpt-3.5-turbo-0125"
            )  # We only use gpt-3.5 to measure token cost
        else:
            class _DummyEncoding:
                def encode(self, s: str):
                    # bytes length as a rough proxy
                    return s.encode("utf-8")
            self.encoding = _DummyEncoding()
        self.temperature = temperature
        self.systemRole = system_role
        self.logger = logger
        return

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

    def run_with_timeout(self, func, timeout):
        """Run a function with timeout that works in multiple threads"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                ("Operation timed out")
                return ""
            except Exception as e:
                self.logger.print_log(f"Operation failed: {e}")
                return ""

    def infer_with_gemini(self, message: str) -> str:
        """Infer using the Gemini model from Google Generative AI"""
        if genai is None:
            self.logger.print_log("Gemini SDK not installed")
            return ""
        gemini_model = genai.GenerativeModel("gemini-pro")

        def call_api():
            message_with_role = self.systemRole + "\n" + message
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                # ...existing safety settings...
            ]
            response = gemini_model.generate_content(
                message_with_role,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                ),
            )
            return response.text

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=50)
                if output:
                    self.logger.print_log("Inference succeeded...")
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""

    def infer_with_openai_model(self, message):
        """Infer using the OpenAI model"""
        if OpenAI is None:
            self.logger.print_log("OpenAI SDK not installed")
            return ""
        api_key = os.environ.get("OPENAI_API_KEY", "").split(":")[0]
        model_input = [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": message},
        ]

        def call_api():
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.online_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=100)
                if output:
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""

    def infer_with_o3_mini_model(self, message):
        """Infer using the o3-mini model"""
        if OpenAI is None:
            self.logger.print_log("OpenAI SDK not installed")
            return ""
        api_key = os.environ.get("OPENAI_API_KEY", "").split(":")[0]
        model_input = [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": message},
        ]

        def call_api():
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.online_model_name, messages=model_input
            )
            return response.choices[0].message.content

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=100)
                if output:
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""

    def infer_with_deepseek_model(self, message):
        """
        Infer using the DeepSeek model
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY2")
        model_input = [
            {
                "role": "system",
                "content": self.systemRole,
            },
            {"role": "user", "content": message},
        ]

        def call_api():
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=self.online_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=300)
                if output:
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""

    def infer_with_claude(self, message):
        """Infer using the Claude model via AWS Bedrock"""
        if boto3 is None or Config is None:
            self.logger.print_log("boto3/botocore not installed for Claude via Bedrock")
            return ""
        if "3.5" in self.online_model_name:
            model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        if "3.7" in self.online_model_name:
            model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

        model_input = [
            {
                "role": "assistant",
                "content": self.systemRole,
            },
            {"role": "user", "content": message},
        ]

        body = json.dumps(
            {
                "messages": model_input,
                "max_tokens": 4000,
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": self.temperature,
                "top_k": 50,
            }
        )

        def call_api():
            client = boto3.client(
                "bedrock-runtime",
                region_name="us-west-2",
                config=Config(read_timeout=100),
            )

            response = (
                client.invoke_model(
                    modelId=model_id, contentType="application/json", body=body
                )["body"]
                .read()
                .decode("utf-8")
            )

            response = json.loads(response)
            return response["content"][0]["text"]

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=100)
                if output:
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {str(e)}")
            time.sleep(2)

        return ""


    def infer_with_glm_model(self, message):
        """Infer using the GLM model"""
        if ZhipuAI is None:
            self.logger.print_log("ZhipuAI SDK not installed")
            return ""
        api_key = os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY")
        model_input = [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": message},
        ]

        def call_api():
            client = ZhipuAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.online_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        tryCnt = 0
        while tryCnt < 5:
            tryCnt += 1
            try:
                output = self.run_with_timeout(call_api, timeout=100)
                if output:
                    # print("Raw response from GLM model: ", output)
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""
