"""
Test script to probe and discover the maximum actual input context length (max_tokens)
supported by each model endpoint using binary search probing.
"""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
API_KEY = os.getenv("LITELLM_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def probe_max_context(model_id: str, max_search: int = 128000) -> dict[str, Any]:
    """
    Finds the maximum input context length (in approximate tokens) supported by a model
    using exponential upper-bound finding followed by binary search refinement.
    """
    # Skip embedding, reranker, and default models
    if any(k in model_id.lower() for k in ("embed", "rerank", "default")):
        return {
            "model": model_id,
            "status": "SKIPPED",
            "max_tokens": None,
            "error": "Non-chat model (embedding / reranker)",
        }

    best_pass = 0
    high = max_search
    curr = 1024
    last_error = None

    # Step 1: Exponential upper-bound search (1k -> 2k -> 4k -> 8k -> 16k -> 32k -> ...)
    while curr <= high:
        prompt = "word " * curr  # ~1 token per "word "
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
            )
            best_pass = curr
            curr *= 2
        except Exception as e:
            last_error = str(e)
            high = curr
            break

    if best_pass == 0:
        return {
            "model": model_id,
            "status": "FAILED",
            "max_tokens": None,
            "error": f"Failed even at min prompt length (1024 words): {last_error}",
        }

    # Step 2: Binary search refinement between best_pass and high
    low = best_pass
    step_precision = 256  # Precision threshold for binary search

    while (high - low) > step_precision:
        mid = (low + high) // 2
        prompt = "word " * mid
        try:
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
            )
            low = mid
            best_pass = mid
        except Exception as e:
            last_error = str(e)
            high = mid

    return {
        "model": model_id,
        "status": "SUPPORTED",
        "max_tokens": best_pass,
        "error": None,
    }
