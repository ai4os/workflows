"""
Propagate Keycloak groups to LiteLLM teams
"""

import logging
import os
import sys
from collections import defaultdict
from urllib.parse import quote

from dotenv import load_dotenv
import requests


load_dotenv()
logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
# logging.basicConfig(level="DEBUG", format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


PAGE_SIZE = 100
LITELLM_URL = os.getenv("LITELLM_BASE_URL", "https://vllm.cloud.ai4eosc.eu").rstrip("/").removesuffix("/v1")
ROOT = "Platform Access"
DRY_RUN = False  # set to False to actually modify LiteLLM
LEVELS = ["ap-a", "ap-a1", "ap-b", "ap-u", "ap-d"]  # low to high


def url_part(value):
    return quote(str(value), safe="")


def normalize(name):
    """'Access-AP-A' and 'ap-a' both become 'ap-a'."""
    name = name.strip().lower()
    return name[len("access-"):] if name.startswith("access-") else name


def highest(levels):
    return max(levels, key=LEVELS.index)


class KeycloakRCIAM:
    """Minimal client for the Keycloak Admin API and the RCIAM group-admin extension."""

    def __init__(self):
        for name in ("KC_URL", "KC_REALM", "KC_CLIENT_ID", "KC_CLIENT_SECRET"):
            if not os.getenv(name):
                raise ValueError(f"Missing {name}")
        base = os.environ["KC_URL"].rstrip("/")
        realm = url_part(os.environ["KC_REALM"])
        self.admin_url = f"{base}/admin/realms/{realm}"
        self.rciam_url = f"{base}/realms/{realm}/agm/account/group-admin"
        self.http = requests.Session()
        self.http.headers["Accept"] = "application/json"

        response = requests.post(
            f"{base}/realms/{realm}/protocol/openid-connect/token",
            data={"grant_type": "client_credentials",
                  "client_id": os.environ["KC_CLIENT_ID"],
                  "client_secret": os.environ["KC_CLIENT_SECRET"]},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Authentication: HTTP {response.status_code}")
        self.http.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    def get(self, url, **kwargs):
        try:
            response = self.http.get(url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"GET {url}: {exc}") from exc
        if not response.ok:
            raise RuntimeError(f"GET {url}: HTTP {response.status_code}: {response.text[:700]}")
        return response.json()

    def find_root(self):
        """Find the top-level ROOT group."""
        groups = self.get(f"{self.admin_url}/groups", params={
            "search": ROOT, "exact": "true", "first": 0, "max": PAGE_SIZE,
        })
        found = [g for g in groups if g.get("path") == f"/{ROOT}"]
        if len(found) != 1:
            raise RuntimeError(f"Expected exactly one group '{ROOT}'; found {len(found)}")
        return found[0]

    def children(self, group_id):
        rows, first = [], 0
        url = f"{self.admin_url}/groups/{url_part(group_id)}/children"
        while True:
            page = self.get(url, params={"first": first, "max": PAGE_SIZE})
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            first += len(page)

    def memberships(self, group_id):
        """Direct RCIAM memberships of a group."""
        items_all, first = [], 0
        url = f"{self.rciam_url}/group/{url_part(group_id)}/members"
        while True:
            page = self.get(url, params={"first": first, "max": PAGE_SIZE, "direct": "true"})
            if not isinstance(page, dict) or not isinstance(page.get("results"), list):
                raise RuntimeError(f"Unexpected RCIAM memberships format: {page}")
            items = page["results"]
            items_all.extend(items)
            first += len(items)
            if len(items) < PAGE_SIZE or first >= page.get("count", 10**12):
                return items_all


def collect_users(api, group, levels):
    """Map user id -> {username, email, roles} for members holding any of the given roles."""
    items = api.memberships(group["id"])
    log.info("Group %s: %d direct memberships", group["path"], len(items))
    users = {}
    for item in items:
        user = item.get("user") or {}
        if not user.get("id"):
            raise RuntimeError(f"Membership without user: {item}")
        roles = {r for r in item.get("groupRoles") or [] if normalize(r) in levels}
        if roles:
            users[user["id"]] = {"username": user.get("username", ""),
                                 "email": user.get("email", ""), "roles": roles}
    return users


def litellm_session():
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("Missing LITELLM_API_KEY")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"
    return session


def litellm_call(session, method, path, **kwargs):
    response = session.request(method, f"{LITELLM_URL}{path}", timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def litellm_data(session):
    """Return (account user ids, user id -> level teams, level -> team id) from LiteLLM."""
    accounts, page = set(), 1
    while True:
        data = litellm_call(session, "GET", "/user/list",
                            params={"page": page, "page_size": PAGE_SIZE})
        accounts.update(u["user_id"] for u in data.get("users") or [])
        if page >= data.get("total_pages", 1):
            break
        page += 1

    levels, team_by_level = defaultdict(set), {}
    for team in litellm_call(session, "GET", "/team/list"):
        level = normalize(team.get("team_alias") or "")
        if level not in LEVELS:
            continue
        team_by_level[level] = team["team_id"]
        for member in team.get("members_with_roles") or []:
            levels[member["user_id"]].add(level)
    return accounts | set(levels), levels, team_by_level


def migrate_user(session, user_id, email, level, current, team_by_level):
    """Move a user and their API keys to the team of `level`, leaving other level teams."""
    target = team_by_level[level]
    if level not in current:
        litellm_call(session, "POST", "/team/member_add", json={
            "team_id": target,
            "member": {"user_id": user_id, "user_email": email, "role": "user"},
        })

    keys, page = [], 1
    while True:
        data = litellm_call(session, "GET", "/key/list", params={
            "user_id": user_id, "return_full_object": "true", "page": page, "size": PAGE_SIZE,
        })
        keys.extend(data["keys"])
        if page >= data.get("total_pages", 1):
            break
        page += 1

    # Keys must change team before the user leaves the old team, or they get erased.
    for key in keys:
        if key.get("team_id") != target:
            litellm_call(session, "POST", "/key/update",
                         json={"key": key["token"], "team_id": target})

    for old in current - {level}:
        litellm_call(session, "POST", "/team/member_delete",
                     json={"team_id": team_by_level[old], "user_id": user_id})
    return len(keys)


def main():
    api = KeycloakRCIAM()
    keycloak, usernames, emails = {}, {}, {}
    for group in api.children(api.find_root()["id"]):
        try:
            users = collect_users(api, group, LEVELS)
        except RuntimeError as exc:
            log.error("Group %s: %s", group["path"], exc)
            continue
        for user_id, u in sorted(users.items(), key=lambda kv: kv[1]["username"]):
            log.debug("Keycloak user %s %s %s: %s", u["username"], user_id, u["email"],
                      sorted(u["roles"]))
            found = {normalize(r) for r in u["roles"]}
            if user_id in keycloak:
                found.add(keycloak[user_id])
            keycloak[user_id] = highest(found)
            usernames[user_id] = u["username"]
            emails[user_id] = u["email"]
        log.info("%d users in %s", len(users), sorted(LEVELS))

    session = litellm_session()
    accounts, litellm, team_by_level = litellm_data(session)
    log.info("%d LiteLLM accounts, %d Keycloak users with a level", len(accounts), len(keycloak))
    misaligned = []
    for user_id in sorted(accounts):
        expected = {keycloak[user_id]} if user_id in keycloak else set()
        actual = litellm.get(user_id, set())
        if expected != actual:
            misaligned.append(user_id)
            log.debug("Misaligned %s %s: keycloak=%s litellm=%s",
                      usernames.get(user_id, "?"), user_id, sorted(expected), sorted(actual))
    log.info("%d misaligned users", len(misaligned))

    migrated = skipped = failed = 0
    for user_id in misaligned:
        level = keycloak.get(user_id)
        if level not in team_by_level:
            skipped += 1
            log.debug("Skip %s: no Keycloak level or no LiteLLM team for %s", user_id, level)
            continue
        label = f"{usernames[user_id]} ({user_id}): {sorted(litellm.get(user_id, set()))} -> {level}"
        if DRY_RUN:
            migrated += 1
            log.debug("Dry run, would migrate %s", label)
            continue
        try:
            n_keys = migrate_user(session, user_id, emails[user_id], level,
                                  litellm.get(user_id, set()), team_by_level)
            migrated += 1
            log.debug("Migrated %s; keys checked: %d", label, n_keys)
        except requests.RequestException as exc:
            failed += 1
            log.debug("Failed migrating %s: %s", label, exc)
    verb = "to migrate (dry run)" if DRY_RUN else "migrated"
    log.info("%d %s, %d skipped, %d failed", migrated, verb, skipped, failed)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        log.critical("%s", exc)
        sys.exit(1)
