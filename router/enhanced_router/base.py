"""Path constants and environment defaults for ClaudeBrigade.

Every path constant is guaranteed to be a ``pathlib.Path``, even when
overridden via environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return a Path from an env var, falling back to *default*.

    If the env var is set, expand leading ``~/`` and convert to ``Path``.
    """
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


#: Directory for persistent runtime state (SQLite database, etc.).
BRIGADE_STATE_DIR = _env_path(
    "BRIGADE_STATE_DIR",
    Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "claude-brigade",
)

#: Directory for transient cache files (LiteLLM config, MCP temp files, etc.).
BRIGADE_CACHE_DIR = _env_path(
    "BRIGADE_CACHE_DIR",
    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "claude-brigade",
)

#: Directory for user-editable configuration (models.yaml, profiles.yaml, etc.).
BRIGADE_CONFIG_DIR = _env_path(
    "BRIGADE_CONFIG_DIR",
    Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "claude-brigade",
)

#: Full path to the SQLite state database.
DEFAULT_DB_PATH: Path = BRIGADE_STATE_DIR / "state.db"

REGISTRY_HASH_KEY = "registry_config_sha256"