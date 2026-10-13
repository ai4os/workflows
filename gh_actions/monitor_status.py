
"""
Monitor the status of GitHub Actions workflows.
If the last N executions (default N=3) have failed:
  - Open an issue in the repository assigned to the workflow owner.
  - Do not reopen or create duplicates if an issue is already open for this workflow.
If the last executions are not failing:
  - Close any existing open issues created for that workflow.

I do not add the issue to the ai4os project because the code complicates quite a bit
(needs to use GH GraphQL API). In any case, this can be done with a separated dedicated
workflow if needed.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import requests
import yaml

load_dotenv()

ACTIONS_PATH = Path(__file__).parent / "actions.yml"
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

ISSUE_TAG_PREFIX = "<!-- gh-action-monitor"


def load_actions(file_path: Path = ACTIONS_PATH) -> list[dict]:
    """Load action definitions from YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    actions = []
    for action in data.get("actions", []):
        repo = action.get("repo")
        name = action.get("name")
        owner = action.get("owner")
        if repo and name is not None:
            actions.append({
                "repo": repo,
                "name": str(name),
                "owner": owner,
            })
    return actions


def get_workflows(repo: str) -> list[dict]:
    """Retrieve all workflows for a repository handling pagination."""
    workflows = []
    page = 1
    per_page = 100

    while True:
        url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows"
        params = {"per_page": per_page, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"[{repo}] Failed to fetch workflows (HTTP {response.status_code}): {response.text}")
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


def get_recent_workflow_runs(repo: str, workflow_id: int, count: int = 3) -> list[dict]:
    """Fetch the most recent completed runs for a workflow."""
    url = f"{GITHUB_API_URL}/repos/{repo}/actions/workflows/{workflow_id}/runs"
    params = {"status": "completed", "per_page": count, "page": 1}
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        print(f"[{repo}] Failed to fetch runs for workflow {workflow_id} (HTTP {response.status_code}): {response.text}")
        return []

    return response.json().get("workflow_runs", [])


def get_open_issues(repo: str) -> list[dict]:
    """Fetch all open issues in the repository."""
    issues = []
    page = 1
    per_page = 100

    while True:
        url = f"{GITHUB_API_URL}/repos/{repo}/issues"
        params = {"state": "open", "per_page": per_page, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"[{repo}] Failed to fetch open issues (HTTP {response.status_code}): {response.text}")
            break

        current = response.json()
        if not isinstance(current, list):
            break

        # Filter out pull requests
        issues.extend([item for item in current if "pull_request" not in item])

        if len(current) < per_page:
            break

        page += 1

    return issues


def find_existing_issue(open_issues: list[dict], workflow_id: int) -> dict | None:
    """Find an already opened issue created by this monitor for the specified workflow."""
    marker = f"{ISSUE_TAG_PREFIX}:workflow_id={workflow_id} -->"
    for issue in open_issues:
        body = issue.get("body") or ""
        if marker in body:
            return issue
    return None


def open_failure_issue(repo: str, workflow: dict, owner: str | None, runs: list[dict], failure_threshold: int) -> dict | None:
    """Open an issue for a continuously failing workflow."""
    workflow_id = workflow["id"]
    workflow_name = workflow.get("name", str(workflow_id))
    wf_path = workflow.get("path", "")
    marker = f"{ISSUE_TAG_PREFIX}:workflow_id={workflow_id} -->"

    title = f"[🤖 CI Alert] Workflow '{workflow_name}' has failed {failure_threshold} consecutive times"

    run_links = "\n".join(
        f"- Run nº{r.get('run_number')} ({r.get('conclusion')}): {r.get('html_url')}"
        for r in runs
    )
    assignee_mention = f"@{owner}" if owner else "No owner specified"

    body = (
        f"{marker}\n"
        f"### Action Failure Notification\n\n"
        f"The workflow **{workflow_name}** (`{wf_path}`) has failed its last **{failure_threshold}** completed runs.\n\n"
        f"**Responsible / Assigned Owner:** {assignee_mention}\n\n"
        f"**Recent Failed Executions:**\n"
        f"{run_links}\n\n"
        f"---\n"
        f"_This issue was automatically opened by the workflow status monitor._"
    )

    payload = {
        "title": title,
        "body": body,
        "labels": ["bug"],
        "type": "Bug",
    }
    if owner:
        payload["assignees"] = [owner]

    url = f"{GITHUB_API_URL}/repos/{repo}/issues"
    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code == 422 and "type" in payload:
        # Fallback if issue types are not enabled in the repository
        fallback_payload = dict(payload)
        fallback_payload.pop("type", None)
        response = requests.post(url, headers=HEADERS, json=fallback_payload)

    if response.status_code == 201:
        issue_data = response.json()
        print(f"🚨 [{repo}] Opened issue #{issue_data.get('number')} for workflow '{workflow_name}' (Assigned to: {owner})")
        return issue_data
    else:
        print(f"[{repo}] Failed to create issue for workflow '{workflow_name}' (HTTP {response.status_code}): {response.text}")
        return None


def close_issue(repo: str, issue: dict, workflow_name: str) -> bool:
    """Close an open monitor issue once the workflow passes again."""
    issue_number = issue["number"]
    url = f"{GITHUB_API_URL}/repos/{repo}/issues/{issue_number}"
    payload = {
        "state": "closed",
        "state_reason": "completed",
    }

    # Add a closing comment explaining why it was closed
    comment_url = f"{GITHUB_API_URL}/repos/{repo}/issues/{issue_number}/comments"
    comment_body = {
        "body": f"Workflow **{workflow_name}** has recovered and is no longer failing. Automatically closing this issue."
    }
    requests.post(comment_url, headers=HEADERS, json=comment_body)

    response = requests.patch(url, headers=HEADERS, json=payload)
    if response.status_code == 200:
        print(f"🎉 [{repo}] Closed issue #{issue_number} because workflow '{workflow_name}' recovered.")
        return True
    else:
        print(f"[{repo}] Failed to close issue #{issue_number} (HTTP {response.status_code}): {response.text}")
        return False


def monitor_status(failure_threshold: int = 3):
    """Monitor workflows across configured repos and manage failure issues."""
    if not GITHUB_TOKEN:
        raise ValueError("You need a valid GITHUB_TOKEN / GH_TOKEN to check workflows and manage issues.")

    actions = load_actions()
    if not actions:
        print("No actions defined in the actions configuration file.")
        return

    # Cache open issues per repo to avoid redundant API queries
    repo_open_issues: dict[str, list[dict]] = {}

    for item in actions:
        repo = item["repo"]
        target_name = item["name"]
        owner = item.get("owner")

        print(f"\nChecking repository '{repo}' for workflow '{target_name}'...")
        workflows = get_workflows(repo)
        if not workflows:
            print(f"🔴 [{repo}] No workflows found or unable to access repository.")
            continue

        if repo not in repo_open_issues:
            repo_open_issues[repo] = get_open_issues(repo)

        matched_workflows = [wf for wf in workflows if workflow_matches(wf, target_name)]
        if not matched_workflows:
            print(f"🔴 [{repo}] No workflow matched '{target_name}'.")
            continue

        for wf in matched_workflows:
            wf_id = wf["id"]
            wf_name = wf.get("name", str(wf_id))

            runs = get_recent_workflow_runs(repo, wf_id, count=failure_threshold)
            if len(runs) < failure_threshold:
                print(f"ℹ️ [{repo}] Workflow '{wf_name}' has only {len(runs)} completed run(s). Threshold is {failure_threshold}.")
                continue

            last_conclusions = [r.get("conclusion") for r in runs]
            all_failed = all(c == "failure" for c in last_conclusions)

            existing_issue = find_existing_issue(repo_open_issues[repo], wf_id)

            if all_failed:
                print(f"❌ [{repo}] Workflow '{wf_name}' failed last {failure_threshold} runs ({last_conclusions}).")
                if existing_issue:
                    print(f"⚠️ [{repo}] Issue #{existing_issue.get('number')} is already open for '{wf_name}'. Not reopening.")
                else:
                    new_issue = open_failure_issue(repo, wf, owner, runs, failure_threshold)
                    if new_issue:
                        repo_open_issues[repo].append(new_issue)
            else:
                print(f"✅ [{repo}] Workflow '{wf_name}' is healthy (conclusions: {last_conclusions}).")
                if existing_issue and close_issue(repo, existing_issue, wf_name):
                    repo_open_issues[repo].remove(existing_issue)


if __name__ == "__main__":
    monitor_status()