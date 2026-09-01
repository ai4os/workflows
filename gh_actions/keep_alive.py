"""
Avoid marking Github Workflows as disabled due to 60-day repo inactivity.
We go over workflows and reenable them if disabled.
"""

import os

from dotenv import load_dotenv
import requests

load_dotenv()

# Workflows that should keep running even if no commits are made to the repo
# Syntax: (repo, workflow_name)
# If workflow name is "*" then keep all repo workflows alive
WORKFLOWS = [
    ("ai4os/ai4os-ai4life-loader", "filter_models"),
]

# Always enable this workflow himself, to avoid being himself shutdown
SELF = ("ai4os/workflows", "keep-alive")

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("You need a valid Github PAT to reeenable workflows.")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def get_workflows(repo: str) -> list[dict]:
    """Retrieve all workflows for a repository, handling pagination."""
    workflows = []
    page = 1
    per_page = 100

    while True:
        url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows"
        params = {"per_page": per_page, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(
                f"[{repo}] Failed to fetch workflows (HTTP {response.status_code}): {response.text}"
            )
            break

        data = response.json()
        current_workflows = data.get("workflows", [])
        workflows.extend(current_workflows)

        if len(current_workflows) < per_page or len(workflows) >= data.get("total_count", 0):
            break

        page += 1

    return workflows


def workflow_matches(workflow: dict, target_name: str) -> bool:
    """Check if a workflow matches the target name, filename, path, or wildcard."""
    if target_name == "*":
        return True

    wf_name = workflow.get("name", "")
    wf_path = workflow.get("path", "")
    wf_filename = wf_path.split("/")[-1] if wf_path else ""
    wf_stem = wf_filename.rsplit(".", 1)[0] if "." in wf_filename else wf_filename

    return target_name in (wf_name, wf_path, wf_filename, wf_stem)


def enable_workflow(repo: str, workflow: dict) -> bool:
    """Reenable a disabled workflow."""
    workflow_id = workflow["id"]
    workflow_name = workflow.get("name", workflow_id)
    url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows/{workflow_id}/enable"

    response = requests.put(url, headers=HEADERS)
    if response.status_code == 204:
        print(f"[{repo}] Successfully re-enabled workflow: '{workflow_name}' (ID: {workflow_id})")
        return True
    else:
        print(
            f"[{repo}] Failed to re-enable workflow '{workflow_name}' "
            f"(HTTP {response.status_code}): {response.text}"
        )
        return False


def keep_alive():
    """Iterate through target workflows and reenable any disabled ones."""
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN / GH_TOKEN environment variable is not set. API calls may fail or be rate-limited.")

    targets = list(WORKFLOWS)
    if SELF not in targets:
        targets.append(SELF)

    for repo, target_name in targets:
        print(f"\nChecking workflows in '{repo}' for '{target_name}'...")
        workflows = get_workflows(repo)
        if not workflows:
            print(f"🔴 [{repo}] No workflows found or unable to access repository.")
            continue

        matched_any = False
        for wf in workflows:
            if workflow_matches(wf, target_name):
                matched_any = True
                wf_name = wf.get("name", wf.get("id"))
                state = wf.get("state", "unknown")

                if state != "active":
                    print(f"🔁 [{repo}] Workflow '{wf_name}' is currently '{state}'. Re-enabling...")
                    enable_workflow(repo, wf)
                else:
                    print(f"✅ [{repo}] Workflow '{wf_name}' is already active.")

        if not matched_any:
            print(f"🔴 [{repo}] No workflow matched '{target_name}'. Available workflows: {[w.get('name') for w in workflows]}")


if __name__ == "__main__":
    keep_alive()
