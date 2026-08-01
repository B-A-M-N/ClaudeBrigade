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

# ---------------------------------------------------------------------------
# Authoritative agent-type allowlists.
#
# These three frozensets are the single source of truth for every hook that
# needs to know which subagent types, mutators, and implementation agents
# exist.  The hooks import them directly -- no lazy import, no fallback.
# ---------------------------------------------------------------------------

#: Agent types allowed to be spawned as subagents.
ALLOWED_SUBAGENTS: frozenset[str] = frozenset({
    "brigade-recon",
    "brigade-implementer",
    "brigade-adversary",
    "brigade-repairer",
    "controller-direct",
    "brigade-fi-qwen-scout",
    "brigade-fi-minimax-architect",
    "brigade-fi-kimi-implementer",
    "brigade-fi-glm-adversary",
    "brigade-fi-glm-fast-repairer",
})

#: Agent types allowed to mutate the workspace
#: (Write / Edit / NotebookEdit / Bash-with-mutation).
MUTATORS: frozenset[str] = frozenset({
    "brigade-implementer",
    "brigade-repairer",
    "controller-direct",
    "brigade-fi-kimi-implementer",
    "brigade-fi-glm-fast-repairer",
})

#: Agent types considered "implementation" for completion-sequence validation.
IMPLEMENTATION_AGENTS: frozenset[str] = frozenset({
    "brigade-implementer",
    "brigade-repairer",
    "controller-direct",
    "brigade-fi-kimi-implementer",
    "brigade-fi-glm-fast-repairer",
})


def _registry_agent_manifest() -> dict[str, dict[str, str]]:
    """Load configured specialist identities when a hook sees one.

    Static role names remain available during bootstrap and when the registry
    is unavailable. Dynamic names are only authorized after the registry has
    described them, so a typo cannot become an implicit agent capability.
    """
    try:
        from enhanced_router.registry import get_registry

        return get_registry().specialist_manifest()
    except Exception:
        return {}


def authorized_subagents() -> frozenset[str]:
    return ALLOWED_SUBAGENTS | frozenset(_registry_agent_manifest())


def mutating_agents() -> frozenset[str]:
    dynamic = {
        name for name, entry in _registry_agent_manifest().items()
        if entry.get("role") in {"implementer", "repairer", "controller"}
    }
    return MUTATORS | frozenset(dynamic)


def implementation_agents() -> frozenset[str]:
    dynamic = {
        name for name, entry in _registry_agent_manifest().items()
        if entry.get("role") in {"implementer", "repairer", "controller"}
    }
    return IMPLEMENTATION_AGENTS | frozenset(dynamic)


def agent_role(agent_type: str) -> str:
    """Resolve a native name's role, including registry-generated names."""
    if agent_type == "controller-direct":
        return "controller"
    entry = _registry_agent_manifest().get(agent_type)
    if entry is not None:
        return entry["role"]
    return next(
        (candidate for candidate in ("recon", "implementer", "adversary", "repairer")
         if candidate in agent_type),
        "recon",
    )

#: HTTP headers that MUST NOT be forwarded to upstream backends (RFC 9113 ?8.2.2).
HOP_BY_HOP: frozenset[str] = frozenset({
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade",
})
