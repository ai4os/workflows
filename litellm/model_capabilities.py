"""
Script to discover models with missing/unconfigured (N/A) capabilities in LiteLLM,
execute corresponding capability tests for missing fields, and upload updated metadata.

Pipeline:
1. Fetch models and metadata info from LiteLLM (/model/info or /model_group/info).
2. Identify models with missing/N/A capabilities (vision, reasoning, tool calling, max input tokens).
3. Execute relevant capability tests for missing capabilities.
4. Save test results to results.yaml.
5. Upload updated metadata back to LiteLLM via PATCH /model/{db_model_id}/update.
"""

import os
from typing import Any

from dotenv import load_dotenv
import requests

from test_capabilities.test_models_maxlen import probe_max_context
from test_capabilities.test_models_reasoning import test_model_reasoning
from test_capabilities.test_models_tool_calling import test_model_tool_calling
from test_capabilities.test_models_vision import test_model_vision


load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

SERVER_ROOT_URL = LITELLM_BASE_URL.rstrip("/").removesuffix("/v1")

# Exclude models containing any of these keywords in their name
EXCLUDE_CRITERIA = [
    "embedding",
    "rerank",
]

# Create persistent session
headers = {
    "Authorization": f"Bearer {LITELLM_API_KEY}",
    "Content-Type": "application/json",
}


def identify_missing_capabilities(model_info: dict[str, Any]) -> list[str]:
    """
    Check which capabilities are missing or N/A (None) in the model's metadata.
    Returns a list of capability keys: ['vision', 'reasoning', 'max_input_tokens', 'tool_calling']
    """
    missing = []
    if model_info.get("supports_vision") is None:
        missing.append("vision")
    if model_info.get("supports_reasoning") is None:
        missing.append("reasoning")
    if model_info.get("supports_function_calling") is None:
        missing.append("tool_calling")

    max_tokens = (
        model_info.get("max_input_tokens")
        or model_info.get("max_tokens")
        or model_info.get("max_model_len")
    )
    if max_tokens is None:
        missing.append("max_input_tokens")

    return missing


def update_model_metadata_in_litellm(
    model_id: str, db_model_id: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Sends a PATCH request to LiteLLM /model/{model_id}/update to partially update
    model_info metadata for a DB-stored model. model_info is merged server-side,
    so only the fields we want to change need to be sent.
    """
    payload = {"model_info": metadata}

    try:
        response = requests.patch(
            f"{SERVER_ROOT_URL}/model/{db_model_id}/update",
            json=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        return {
            "model": model_id,
            "success": True,
            "response": response.json(),
            "error": None,
        }
    except requests.exceptions.RequestException as e:
        err_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                err_msg = f"{e.response.status_code} {e.response.reason} - {e.response.json()}"
            except (ValueError, KeyError, AttributeError):
                err_msg = (
                    f"{e.response.status_code} {e.response.reason} - {e.response.text}"
                )
        return {
            "model": model_id,
            "success": False,
            "response": None,
            "error": err_msg,
        }


def run_tests_for_missing_capabilities(
    model_name: str, missing_caps: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run tests only for the capabilities that are missing/N/A.
    Returns:
      - test_records: dict of test results for yaml persistence
      - new_metadata: dict of updated metadata values to send to LiteLLM
    """
    test_records: dict[str, Any] = {}
    new_metadata: dict[str, Any] = {}

    if "vision" in missing_caps:
        print("  [1/4] Probing Vision ... ", end="", flush=True)
        res = test_model_vision(model_name)
        v_icon = (
            "✅"
            if res["status"] == "SUPPORTED"
            else ("⏭️" if res["status"] == "SKIPPED" else "❌")
        )
        print(f"{v_icon} {res['status']}")
        test_records["vision"] = res
        new_metadata["supports_vision"] = res.get("status") == "SUPPORTED"

    if "reasoning" in missing_caps:
        print("  [2/4] Probing Reasoning ... ", end="", flush=True)
        res = test_model_reasoning(model_name)
        r_icon = (
            "✅"
            if res["status"] == "SUPPORTED"
            else ("⏭️" if res["status"] == "SKIPPED" else "❌")
        )
        print(f"{r_icon} {res['status']}")
        test_records["reasoning"] = res
        new_metadata["supports_reasoning"] = res.get("status") == "SUPPORTED"

    if "tool_calling" in missing_caps:
        print("  [3/4] Probing Tool Calling ... ", end="", flush=True)
        res = test_model_tool_calling(model_name)
        t_icon = (
            "✅"
            if res["status"] == "SUPPORTED"
            else ("⏭️" if res["status"] == "SKIPPED" else "❌")
        )
        print(f"{t_icon} {res['status']}")
        test_records["multi_turn_tool_calling"] = res
        new_metadata["supports_function_calling"] = res.get("status") == "SUPPORTED"

    if "max_input_tokens" in missing_caps:
        print("  [4/4] Probing Max Input Context ... ", end="", flush=True)
        res = probe_max_context(model_name)
        m_icon = (
            "✅"
            if res["status"] == "SUPPORTED"
            else ("⏭️" if res["status"] == "SKIPPED" else "❌")
        )
        max_str = f"~{res['max_tokens']:,} tokens" if res.get("max_tokens") else "N/A"
        print(f"{m_icon} {res['status']} ({max_str})")
        test_records["max_input_context"] = res
        if res.get("max_tokens") is not None:
            new_metadata["max_input_tokens"] = int(res["max_tokens"])

    return test_records, new_metadata


def main():
    print(f"Connecting to LiteLLM server: {SERVER_ROOT_URL}")

    # Retrieve models from LiteLLM
    response = requests.get(
        f"{SERVER_ROOT_URL}/model/info", headers=headers, timeout=10
    )
    response.raise_for_status()
    models_data = response.json().get("data", [])

    if not models_data:
        raise Exception("❌ No models retrieved from LiteLLM.")

    print(
        f"Retrieved metadata for {len(models_data)} model(s). Scanning for unconfigured capabilities...\n"
    )

    models_to_update = []
    for item in models_data:
        model_name = item.get("model_name") or item.get("model_group") or item.get("id")
        if not model_name:
            continue
        if any(
            criterion.lower() in model_name.lower() for criterion in EXCLUDE_CRITERIA
        ):
            continue
        info = item.get("model_info") or item
        missing_caps = identify_missing_capabilities(info)
        if missing_caps:
            models_to_update.append((model_name, item, info, missing_caps))

    if not models_to_update:
        print(
            "✨ All models already have full capabilities metadata configured in LiteLLM."
        )
        return

    print(f"Found {len(models_to_update)} model(s) with missing / N/A capabilities:\n")
    for name, _, _, caps in models_to_update:
        print(f"  • {name}: missing {caps}")
    print()

    # Filter out non healthy models
    response = requests.get(
        f"{SERVER_ROOT_URL}/health/latest", headers=headers, timeout=10
    )
    response.raise_for_status()
    health_data = response.json().get("latest_health_checks", {})
    unhealthy_models = {
        check.get("model_name")
        for check in health_data.values()
        if isinstance(check, dict) and check.get("status") == "unhealthy"
    }

    for model_name, item, info, missing_caps in models_to_update:
        if model_name in unhealthy_models:
            print(f"\n⏭️  Skipping unhealthy model: [{model_name}]")
            continue

        print(f"\n{'=' * 80}")
        print(f"🔍 Testing & Updating: [{model_name}]")
        print(f"   Missing capabilities: {missing_caps}")
        print(f"{'=' * 80}")

        test_records, new_metadata = run_tests_for_missing_capabilities(
            model_name, missing_caps
        )

        # Upload metadata to LiteLLM if applicable
        server_info = item.get("model_info", {})
        is_db_model = server_info.get("db_model", False)
        db_model_id = server_info.get("id")

        print(f"\n📤 Updating LiteLLM metadata for [{model_name}] ...")
        print(f"   Payload: {new_metadata}")

        if not is_db_model or not db_model_id:
            print(
                "   ⏭️  Skipped uploading: model is statically configured (db_model: false) or has no DB id."
            )
            print("       Edit the server config.yaml directly to set these values.")
            continue

        res = update_model_metadata_in_litellm(model_name, db_model_id, new_metadata)
        if res.get("success"):
            print("   ✅ Successfully updated model metadata in LiteLLM!")
        else:
            raise Exception(f"   ❌ Failed to update LiteLLM: {res.get('error')}")


if __name__ == "__main__":
    main()
