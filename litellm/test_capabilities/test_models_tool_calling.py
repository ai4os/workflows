"""
Test script to inspect all models available on the endpoint and verify
which ones support multi-turn function/tool calling (tool_calls -> tool response -> final synthesis).
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu/v1")
API_KEY = os.getenv("LITELLM_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Dummy tool definition to test function calling capability
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. London",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def test_model_tool_calling(model_id: str) -> dict[str, Any]:
    """
    Test a 2-turn tool calling workflow on a single model:
      Turn 1: Model should generate a tool_call.
      Turn 2: Model should accept the tool execution response and produce a final answer.
    """
    messages = [
        {
            "role": "user",
            "content": "What is the weather in Paris? Use the provided tool.",
        }
    ]

    # Skip embedding models by name heuristic
    if "embed" in model_id.lower():
        return {
            "model": model_id,
            "status": "SKIPPED",
            "turn_1_tool_call": False,
            "turn_2_tool_response": False,
            "error": "Embedding model (not chat/completion)",
        }

    try:
        # Turn 1: Ask model to use the tool
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=SAMPLE_TOOLS,
            tool_choice="auto",
        )
    except Exception as e:
        return {
            "model": model_id,
            "status": "FAILED_TURN_1",
            "turn_1_tool_call": False,
            "turn_2_tool_response": False,
            "error": f"Turn 1 error: {e}",
        }

    msg = response.choices[0].message
    if not msg.tool_calls:
        return {
            "model": model_id,
            "status": "NO_TOOL_CALL_TRIGGERED",
            "turn_1_tool_call": False,
            "turn_2_tool_response": False,
            "error": f"Model returned plain text instead of tool call: {msg.content!r}",
        }

    # Turn 2: Provide tool output back to the model
    tool_call = msg.tool_calls[0]
    messages.append(msg.model_dump())
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                {"city": "Paris", "temperature": "18C", "condition": "Sunny"}
            ),
        }
    )

    try:
        final_response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=SAMPLE_TOOLS,
        )
        final_text = final_response.choices[0].message.content
        return {
            "model": model_id,
            "status": "SUPPORTED",
            "turn_1_tool_call": True,
            "turn_2_tool_response": True,
            "error": None,
            "sample_output": (final_text or "").strip()[:100],
        }
    except Exception as e:
        return {
            "model": model_id,
            "status": "FAILED_TURN_2",
            "turn_1_tool_call": True,
            "turn_2_tool_response": False,
            "error": f"Turn 2 error: {e}",
        }
