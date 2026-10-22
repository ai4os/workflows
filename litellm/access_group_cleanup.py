"""
Clean up LiteLLM access groups.

Iterate through access groups and remove models, agents, and MCP servers
that are listed in an access group but do not actually exist in the LiteLLM deployment.

Ideally when one deletes an asset in LiteLLM, LiteLLM should also delete the asset from
the access group. But that does not seem to be the current behavior:
  https://github.com/BerriAI/litellm/issues/45573
"""

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


def get_available_models() -> set[str]:
    """Retrieve all available model names from LiteLLM (/model/info)."""
    response = session.get(f"{SERVER_ROOT_URL}/model/info")
    response.raise_for_status()
    models_data = response.json().get("data", [])
    model_names = {
        m["model_name"]
        for m in models_data
        if isinstance(m, dict) and m.get("model_name")
    }
    return model_names


def get_available_agents() -> set[str]:
    """Retrieve all available agent IDs from LiteLLM (/v1/agents)."""
    response = session.get(f"{SERVER_ROOT_URL}/v1/agents")
    response.raise_for_status()
    agents_data = response.json()
    agent_ids = {
        a["agent_id"]
        for a in agents_data
        if isinstance(a, dict) and a.get("agent_id")
    }
    return agent_ids


def get_available_mcp_servers() -> set[str]:
    """Retrieve all available MCP server IDs from LiteLLM (/v1/mcp/server)."""
    response = session.get(f"{SERVER_ROOT_URL}/v1/mcp/server")
    response.raise_for_status()
    mcp_data = response.json()
    server_ids = {
        s["server_id"]
        for s in mcp_data
        if isinstance(s, dict) and s.get("server_id")
    }
    return server_ids


def get_access_groups() -> list[dict[str, Any]]:
    """Retrieve all access groups from LiteLLM (/v1/access_group)."""
    response = session.get(f"{SERVER_ROOT_URL}/v1/access_group")
    response.raise_for_status()
    return response.json()


def update_access_group(access_group_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Update an access group via PUT /v1/access_group/{access_group_id}."""
    response = session.put(
        f"{SERVER_ROOT_URL}/v1/access_group/{access_group_id}",
        json=payload,
    )
    response.raise_for_status()
    return response.json()


def cleanup_access_groups(dry_run: bool = False) -> None:
    """
    Find and remove non-existent models, agents, and MCP servers from access groups.

    :param dry_run: If True, only log changes without updating access groups.
    """
    print("🔍 Fetching available resources from LiteLLM...")
    available_models = get_available_models()
    available_agents = get_available_agents()
    available_mcps = get_available_mcp_servers()

    print(f"  • Models: {len(available_models)}")
    print(f"  • Agents: {len(available_agents)}")
    print(f"  • MCP servers: {len(available_mcps)}")

    print("\n📋 Fetching access groups...")
    access_groups = get_access_groups()
    print(f"Found {len(access_groups)} access group(s).\n")

    for ag in access_groups:
        ag_id = ag.get("access_group_id")
        ag_name = ag.get("access_group_name", "<unnamed>")
        print(f"=== Checking access group: '{ag_name}' ({ag_id}) ===")

        current_models = ag.get("access_model_names") or []
        current_agents = ag.get("access_agent_ids") or []
        current_mcps = ag.get("access_mcp_server_ids") or []

        # Filter to preserve order while keeping only existing items
        valid_models = [m for m in current_models if m in available_models]
        removed_models = [m for m in current_models if m not in available_models]

        valid_agents = [a for a in current_agents if a in available_agents]
        removed_agents = [a for a in current_agents if a not in available_agents]

        valid_mcps = [s for s in current_mcps if s in available_mcps]
        removed_mcps = [s for s in current_mcps if s not in available_mcps]

        has_changes = bool(removed_models or removed_agents or removed_mcps)

        if removed_models:
            print(f"  ❌ Stale models ({len(removed_models)}): {removed_models}")
        if removed_agents:
            print(f"  ❌ Stale agents ({len(removed_agents)}): {removed_agents}")
        if removed_mcps:
            print(f"  ❌ Stale MCP servers ({len(removed_mcps)}): {removed_mcps}")

        if not has_changes:
            print("  ✅ All listed resources exist. No cleanup needed.\n")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would update access group '{ag_name}'.\n")
            continue

        update_payload = {
            "access_group_name": ag.get("access_group_name"),
            "description": ag.get("description"),
            "access_model_names": valid_models,
            "access_mcp_server_ids": valid_mcps,
            "access_agent_ids": valid_agents,
            "assigned_team_ids": ag.get("assigned_team_ids") or [],
            "assigned_key_ids": ag.get("assigned_key_ids") or [],
        }

        print(f"  💾 Updating access group '{ag_name}'...")
        update_access_group(ag_id, update_payload)
        print(f"  ✅ Successfully cleaned up '{ag_name}'.\n")

    print("✨ Access group cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up non-existent models, agents, and MCP servers from LiteLLM access groups."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected changes without modifying access groups in LiteLLM.",
    )
    args = parser.parse_args()
    cleanup_access_groups(dry_run=args.dry_run)
