import argparse
import os
from typing import Any

from dotenv import load_dotenv
import requests

load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

if not LITELLM_API_KEY:
    raise ValueError("LITELLM_API_KEY environment variable is required.")

SERVER_ROOT_URL = LITELLM_BASE_URL.rstrip("/").removesuffix("/v1")

# Create persistent requests session
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {LITELLM_API_KEY}",
    "Content-Type": "application/json",
})


# Make sure aliases haven't stopped working because the underlying model has been removed
ALIASES = [
    "AI4EOSC/default",
    "AI4EOSC/default-agentic",
    "AI4EOSC/default-embeddings",
]
try:
    response = session.get(f"{SERVER_ROOT_URL}/models", timeout=10)
    if response.status_code == 200:
        models = response.json().get("data", [])
        model_ids = [m.get("id") for m in models]
        missing_aliases = [alias for alias in ALIASES if alias not in model_ids]
        if missing_aliases:
            raise ValueError(f"Aliases missing from model list: {', '.join(missing_aliases)}")
    else:
        print(f"Failed to list models (status {response.status_code}): {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Failed to list models: {e}")
