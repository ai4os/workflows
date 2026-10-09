"""
Test script to inspect all models available on the endpoint and verify
which ones support reasoning / thinking capabilities.
"""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
API_KEY = os.getenv("LITELLM_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Common XML/HTML tags and section headers used by models for thinking / chain-of-thought
REASONING_TAGS = [
    "<think>",
    "</think>",
    "<thought>",
    "</thought>",
    "<thinking>",
    "</thinking>",
    "<reasoning>",
    "</reasoning>",
    "<chain_of_thought>",
    "</chain_of_thought>",
    "<thought_process>",
    "</thought_process>",
]

REASONING_HEADERS = [
    "thought:",
    "thinking:",
    "reasoning:",
    "### thinking",
    "### reasoning",
    "### thought",
    "**thinking process:**",
    "**reasoning process:**",
    "**thought process:**",
]


def check_reasoning_in_response(msg: Any) -> tuple[bool, str | None]:
    """
    Check if a chat completion message contains reasoning either via API fields,
    dedicated message attributes, XML tags, or reasoning section headers in content.
    """
    dump = msg.model_dump() if hasattr(msg, "model_dump") else {}

    # Check for dedicated reasoning fields (OpenAI, LiteLLM, DeepSeek, Anthropic, vLLM standards)
    reasoning_content = (
        getattr(msg, "reasoning_content", None)
        or dump.get("reasoning_content")
        or dump.get("reasoning")
        or dump.get("thinking")
        or dump.get("thought")
    )
    if reasoning_content:
        return True, str(reasoning_content)

    content = msg.content or ""
    content_lower = content.lower().strip()

    # Check for reasoning/thinking XML tags in content
    if any(tag in content_lower for tag in REASONING_TAGS):
        return True, content

    # Check for explicit reasoning section headers
    if any(
        content_lower.startswith(hdr) or f"\n{hdr}" in content_lower
        for hdr in REASONING_HEADERS
    ):
        return True, content

    return False, None


def test_model_reasoning(model_id: str) -> dict[str, Any]:
    """
    Test reasoning capability on a single model:
      Attempt 1: System prompt + user prompt requesting step-by-step reasoning.
      Attempt 2: Request with reasoning_effort or framework-specific thinking parameters.
    """
    # Skip non-chat models
    if any(k in model_id.lower() for k in ("embed", "rerank", "default")):
        return {
            "model": model_id,
            "status": "SKIPPED",
            "reasoning_supported": False,
            "error": "Non-chat model (embedding / reranker)",
        }

    # Prompt configuration with system prompt encouraging explicit chain-of-thought
    messages = [
        {
            "role": "system",
            "content": (
                "You are a reasoning assistant. Always output your step-by-step thinking process "
                "before providing the final answer."
            ),
        },
        {
            "role": "user",
            "content": "How many 'r's are in the word 'strawberry'? Think step by step.",
        },
    ]

    # Parameter variations to trigger reasoning/thinking mode across different backends (vLLM, LiteLLM, Ollama, etc.)
    param_variations = [
        {},  # Standard request with system prompt
        {"reasoning_effort": "medium"},  # OpenAI / LiteLLM standard
    ]

    last_content = ""
    for params in param_variations:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=400,
                **params,
            )
            msg = response.choices[0].message
            last_content = msg.content or ""
            is_reasoning, sample = check_reasoning_in_response(msg)

            if is_reasoning:
                return {
                    "model": model_id,
                    "status": "SUPPORTED",
                    "reasoning_supported": True,
                    "error": None,
                    "sample_output": (sample or "").strip()[:100],
                }
        except Exception:
            pass

    return {
        "model": model_id,
        "status": "NO_REASONING_DETECTED",
        "reasoning_supported": False,
        "error": "Response did not contain reasoning_content, thinking tags, or reasoning section headers",
        "sample_output": last_content.strip()[:100],
    }
