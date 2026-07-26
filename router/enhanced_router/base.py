"""Path constants and environment defaults for ClaudeBrigade."""

from __future__ import annotations

import os
from pathlib import Path

BRIGADE_STATE_DIR = os.getenv("BRIGADE_STATE_DIR") or (
    Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "claude-brigade"
)
BRIGADE_CACHE_DIR = os.getenv("BRIGADE_CACHE_DIR") or (
    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "claude-brigade"
)
BRIGADE_CONFIG_DIR = os.getenv("BRIGADE_CONFIG_DIR") or (
    Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "claude-brigade"
)
DEFAULT_DB_PATH = BRIGADE_STATE_DIR / "state.db"
REGISTRY_HASH_KEY = "registry_config_sha256"
