"""
GitHub bus: the glue between the pool connector and the miners.

jobs.txt   (written by pool_connector, read by pearl_miner)
shares.txt (appended by pearl_miner, drained by pool_connector)

Uses the GitHub Contents API (repo must exist; token needs repo write scope).
Concurrent writes are handled with optimistic locking (read sha -> PUT, retry
on 409 conflict).

Also ships a LocalBus for offline testing (plain files).
"""

from __future__ import annotations

import base64
import json
import os
import time

import requests

CONFIG_FILES = ("config.json", os.path.expanduser("~/.pearl_miner.json"))


def get_token_from_env_or_config() -> str | None:
    """Token from GH_TOKEN/GITHUB_TOKEN env, then config.json (git-ignored)."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and token != "REPLACE_ME":
        return token.strip()
    for name in CONFIG_FILES:
        if os.path.exists(name):
            try:
                t = json.load(open(name)).get("github_token")
            except Exception:
                continue
            if t and t != "REPLACE_ME":
                return t.strip()
    return None


def resolve_repo(repo: str, token: str | None) -> str:
    """'pearl-miner' -> '<owner>/pearl-miner' (owner resolved from the token).

    'owner/repo' passes through untouched. Raises a clear error when the token
    is missing or rejected by GitHub (instead of a confusing KeyError).
    """
    if "/" in repo:
        return repo
    if not token:
        raise RuntimeError(
            "GitHub token not found.\n"
            "  Set it with:  export GH_TOKEN=ghp_xxx\n"
            "  or put it in config.json (github_token) — this file is git-ignored.\n"
            "  Or run offline:  --bus local --local-dir /tmp/pearl_bus"
        )
    me = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    ).json()
    if "login" not in me:
        raise RuntimeError(
            f"GitHub auth failed (is the token valid/expired?): "
            f"{me.get('message', me)}"
        )
    return f"{me['login']}/{repo}"


class GitHubBus:
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str, *, max_retries: int = 8):
        if not token or token == "REPLACE_ME":
            raise RuntimeError(
                "GitHub token is required — export GH_TOKEN or set github_token in config.json"
            )
        self.token = token
        self.repo = repo
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    # ------------------------------------------------------------------ #
    def get_file(self, path: str) -> tuple[bytes | None, str | None]:
        """Return (decoded content bytes, blob sha) or (None, None)."""
        url = f"{self.API}/repos/{self.repo}/contents/{path}"
        r = self._session.get(url)
        if r.status_code == 404:
            return None, None
        if r.status_code != 200:
            raise RuntimeError(f"GET {path}: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return base64.b64decode(data["content"]), data["sha"]

    def put_file(self, path: str, content: bytes, *, message: str | None = None) -> str:
        """Create or update a file. Returns the new blob sha."""
        url = f"{self.API}/repos/{self.repo}/contents/{path}"
        for attempt in range(self.max_retries):
            # fetch the current sha fresh on every attempt (avoids stale-sha 409s
            # when multiple writers race)
            _, sha = self.get_file(path)
            payload = {
                "message": message or f"update {path}",
                "content": base64.b64encode(content).decode(),
            }
            if sha:
                payload["sha"] = sha
            r = self._session.put(url, json=payload)
            if r.status_code in (200, 201):
                return r.json()["content"]["sha"]
            if r.status_code == 409 and attempt < self.max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"PUT {path}: HTTP {r.status_code}: {r.text[:400]}")
        raise RuntimeError(f"PUT {path}: out of retries")

    # ------------------------------------------------------------------ #
    def write_text(self, path: str, text: str, *, message: str | None = None) -> str:
        return self.put_file(path, text.encode("utf-8"), message=message)

    def read_text(self, path: str) -> str | None:
        data, _ = self.get_file(path)
        return None if data is None else data.decode("utf-8", errors="replace")

    def append_line(self, path: str, line: str) -> bool:
        """Append a line with conflict-safe retry. True if appended."""
        line = line.rstrip("\n") + "\n"
        for attempt in range(self.max_retries):
            content, sha = self.get_file(path)
            new_content = (content or b"") + line.encode("utf-8")
            try:
                self.put_file(path, new_content, message=f"append to {path}")
                return True
            except RuntimeError as e:
                if "409" in str(e) and attempt < self.max_retries - 1:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise
        return False

    def clear(self, path: str) -> None:
        """Empty the file (used after shares are drained)."""
        self.write_text(path, "", message=f"clear {path}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_env_or_config(repo: str, config: dict | None = None) -> "GitHubBus":
        token = get_token_from_env_or_config()
        if not token:
            raise RuntimeError(
                "GitHub token not found: set GH_TOKEN env var or github_token in config.json"
            )
        return GitHubBus(token, resolve_repo(repo, token))


class LocalBus:
    """Plain-file bus for offline end-to-end tests (no GitHub needed)."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def read_text(self, path: str) -> str | None:
        try:
            with open(self._path(path), "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write_text(self, path: str, text: str):
        with open(self._path(path), "w", encoding="utf-8") as f:
            f.write(text)

    def append_line(self, path: str, line: str) -> bool:
        with open(self._path(path), "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
        return True

    def clear(self, path: str):
        self.write_text(path, "")

    def get_file(self, path: str):
        c = self.read_text(path)
        return (None if c is None else c.encode()), None

    def put_file(self, path: str, content: bytes, **kw):
        self.write_text(path, content.decode("utf-8", errors="replace"))
        return "local"
