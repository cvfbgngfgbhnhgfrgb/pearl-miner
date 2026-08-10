"""
github_sync.py — GitHub synchronization helpers for jobs.txt and shares.txt.

Provides convenience functions wrapping GitHubBus for easy importing.
"""

from __future__ import annotations

import os
from github_bus import GitHubBus, LocalBus, get_token_from_env_or_config, resolve_repo

DEFAULT_REPO = os.environ.get("PEARL_REPO", "pearl-mining")

_default_bus = None


def get_default_bus(repo: str = DEFAULT_REPO):
    global _default_bus
    if _default_bus is None:
        token = get_token_from_env_or_config()
        if token:
            full_repo = resolve_repo(repo, token)
            _default_bus = GitHubBus(token, full_repo)
        else:
            _default_bus = LocalBus("/tmp/pearl_bus")
    return _default_bus


def get_file(path: str, repo: str = DEFAULT_REPO) -> str | None:
    bus = get_default_bus(repo)
    return bus.read_text(path)


def set_file(path: str, content: str, repo: str = DEFAULT_REPO) -> str:
    bus = get_default_bus(repo)
    return bus.write_text(path, content)


def append_file(path: str, line: str, repo: str = DEFAULT_REPO) -> bool:
    bus = get_default_bus(repo)
    return bus.append_line(path, line)


def clear_file(path: str, repo: str = DEFAULT_REPO) -> None:
    bus = get_default_bus(repo)
    bus.clear(path)
