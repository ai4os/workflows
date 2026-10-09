"""
Test script to inspect all models available on the endpoint and verify
which ones support vision / multimodal image inputs.
"""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
API_KEY = os.getenv("LITELLM_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 1x1 transparent PNG data URI as a lightweight image for vision capability probing
TINY_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_model_vision(model_id: str) -> dict[str, Any]:
    """
    Test vision capability on a single model by sending a tiny image payload.
    Uses max_tokens=1 to keep inference fast and avoid unnecessary generation.
    """
    # Skip embedding and rerank models by name heuristic
    if any(k in model_id.lower() for k in ("embed", "rerank")):
        return {
            "model": model_id,
            "status": "SKIPPED",
            "vision_supported": False,
            "error": "Non-chat model (embedding / reranker)",
        }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": TINY_IMAGE_DATA_URL},
                },
            ],
        }
    ]

    try:
        # max_tokens=1 ensures the model stops immediately after 1 token, making execution very fast
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1,
        )
        sample_text = (response.choices[0].message.content or "").strip()
        return {
            "model": model_id,
            "status": "SUPPORTED",
            "vision_supported": True,
            "error": None,
            "sample_output": sample_text[:50],
        }
    except Exception as e:
        err_msg = str(e)
        return {
            "model": model_id,
            "status": "NOT_SUPPORTED",
            "vision_supported": False,
            "error": err_msg,
        }
