"""
Manual model discovery.

We choose to do manual model discovery [1] instead of using LiteLLM's model discovery
because with automated discovery:
* individual models do not appear in the UI (only in the API)
* wildcard appears as a model (this can be filter out from the genai-ui but will still
appear when users query the LiteLLM API)
* we cannot set individual prices for the different provider models
* we cannot run individual health checks

For reference, automated model discovery was created with:
- Model Name: ENGRAMMER/*
- LiteLLM Model Name: openai/*
- Provider: openai

[1]: https://docs.litellm.ai/docs/proxy/model_discovery
"""

import os

import requests
from dotenv import load_dotenv


load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# Before running this script make sure each provider has credentials saved in LiteLLM
# with a matching name
PROVIDER_NAMES = [
    "engrammer",
    "ifca01",
    "iisas01",
]

# Filter out some "models" like IFCA01's 'misc/latency-endpoint'
FILTER_OUT = ["latency"]

# Define some default input/output token costs
INPUT_COST, OUTPUT_COST = 1e-07, 3e-07

for provider_name in PROVIDER_NAMES:
    # Retrieve provider credentials (base URL and masked API key)
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    response = requests.get(
        f"{LITELLM_BASE_URL.rstrip('/')}/credentials/by_name/{provider_name}",
        headers=headers,
    )
    response.raise_for_status()
    cred_data = response.json()

    credential_values = cred_data.get("credential_values", {})
    provider_base_url = credential_values.get("api_base")
    masked_api_key = credential_values.get("api_key")

    # Retrieve full API key from env variables, check lasts digits match with masked one
    env_var_name = f"{provider_name.upper()}_API_KEY"
    full_api_key = os.getenv(env_var_name)
    if not full_api_key:
        raise ValueError(
            f"Environment variable '{env_var_name}' not found for provider '{provider_name}'"
        )

    if masked_api_key:
        masked_suffix = masked_api_key.split("...")[-1].split("*")[-1]
        if not full_api_key.endswith(masked_suffix):
            raise ValueError(
                f"API key in {env_var_name} does not match masked key '{masked_api_key}' "
                f"(expected to end with '{masked_suffix}')"
            )

    # Retrieve models from provider
    provider_headers = {"Authorization": f"Bearer {full_api_key}"}
    provider_models_url = f"{provider_base_url.rstrip('/')}/models"
    try:
        resp = requests.get(provider_models_url, headers=provider_headers)
        if resp.status_code == 404 and not provider_base_url.rstrip("/").endswith("/v1"):
            resp = requests.get(
                f"{provider_base_url.rstrip('/')}/v1/models", headers=provider_headers
            )
        resp.raise_for_status()
    except Exception:
        print(f"❌ Failed to query {provider_name} provider")
        continue
    provider_models_data = resp.json().get("data", [])
    provider_model_ids = {
        m["id"] for m in provider_models_data if isinstance(m, dict) and "id" in m
    }

    # Check if which models exist in LiteLLM
    litellm_models_resp = requests.get(
        f"{LITELLM_BASE_URL.rstrip('/')}/model/info",
        headers=headers,
    )
    litellm_models_resp.raise_for_status()
    litellm_models_data = litellm_models_resp.json().get("data", [])

    # Map existing models in LiteLLM for this provider
    existing_provider_models = {}
    for model_entry in litellm_models_data:
        params = model_entry.get("litellm_params", {})
        if provider_name == params.get("litellm_credential_name"):
            model_id = params.get("model", "").removeprefix("litellm_proxy/")
            litellm_id = model_entry.get("model_info", {}).get("id")
            if model_id:
                existing_provider_models[model_id] = litellm_id

    # Register new models
    models_to_add = provider_model_ids - set(existing_provider_models.keys())
    for model_id in models_to_add:
        if any(f in model_id for f in FILTER_OUT):
            continue

        new_model_data = {
            "model_name": model_id,
            "litellm_params": {
                "model": f"litellm_proxy/{model_id}",
                "litellm_credential_name": provider_name,
                "input_cost_per_token": INPUT_COST,
                "output_cost_per_token": OUTPUT_COST,
            },
            "model_info": {},
        }
        add_resp = requests.post(
            f"{LITELLM_BASE_URL.rstrip('/')}/model/new",
            headers=headers,
            json=new_model_data,
        )
        add_resp.raise_for_status()
        print(f"[{provider_name}] Registered model: {model_id}")

    # Delete old models
    models_to_delete = set(existing_provider_models.keys()) - provider_model_ids
    for model_id in models_to_delete:
        litellm_id = existing_provider_models[model_id]
        if litellm_id:
            del_resp = requests.post(
                f"{LITELLM_BASE_URL.rstrip('/')}/model/delete",
                headers=headers,
                json={"id": litellm_id},
            )
            del_resp.raise_for_status()
            print(f"[{provider_name}] Deleted old model: {model_id}")

    print(f"✅ Processed {provider_name} provider")
