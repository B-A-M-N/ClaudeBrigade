"""Path constants and environment defaults for ClaudeBrigade.

Every path constant is guaranteed to be a ``pathlib.Path``, even when
overridden via environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
})

#: Agent types allowed to mutate the workspace
#: (Write / Edit / NotebookEdit / Bash-with-mutation).
MUTATORS: frozenset[str] = frozenset({
    "brigade-implementer",
    "brigade-repairer",
    "controller-direct",
})

#: Agent types considered "implementation" for completion-sequence validation.
IMPLEMENTATION_AGENTS: frozenset[str] = frozenset({
    "brigade-implementer",
    "brigade-repairer",
    "controller-direct",
})


@dataclass(frozen=True)
class AgentCapabilities:
    """Capability snapshot projected for one native worker identity."""

    roles: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    can_mutate: bool = False
    isolation: str = "none"
    may_spawn_agents: bool = False
    may_integrate: bool = False
    may_adjudicate: bool = False
    counts_as_implementation: bool = False


def _registry_agent_manifest() -> dict[str, dict]:
    """Load configured specialist identities when a hook sees one.

    Static role names remain available during bootstrap and when the registry
    is unavailable. Dynamic names are only authorized after the registry has
    described them, so a typo cannot become an implicit agent capability.
    """
    try:
        from enhanced_router.registry import get_registry

        registry = get_registry()
        if hasattr(registry, "native_worker_manifest"):
            return registry.native_worker_manifest()
        return registry.specialist_manifest()
    except Exception:
        return {}


def authorized_subagents() -> frozenset[str]:
    return ALLOWED_SUBAGENTS | frozenset(_registry_agent_manifest())


def mutating_agents() -> frozenset[str]:
    dynamic = {
        name for name, entry in _registry_agent_manifest().items()
        if entry.get("can_mutate") is True
    }
    return MUTATORS | frozenset(dynamic)


def implementation_agents() -> frozenset[str]:
    dynamic = {
        name for name, entry in _registry_agent_manifest().items()
        if entry.get("counts_as_implementation") is True
    }
    return IMPLEMENTATION_AGENTS | frozenset(dynamic)


def agent_capabilities(agent_type: str) -> AgentCapabilities:
    """Return the explicit capability policy for a native worker.

    Unknown identities are read-only. Role-name substring matching is kept
    only in ``agent_role`` for diagnostic compatibility and is never used as
    the mutation authority boundary.
    """
    entry = _registry_agent_manifest().get(agent_type)
    if entry is not None:
        roles = tuple(str(item) for item in (entry.get("roles") or [entry.get("role", "recon")]))
        return AgentCapabilities(
            roles=roles,
            tools=tuple(str(item) for item in entry.get("tools", ())),
            disallowed_tools=tuple(str(item) for item in entry.get("disallowed_tools", ())),
            can_mutate=bool(entry.get("can_mutate")),
            isolation=str(entry.get("isolation") or "none"),
            may_spawn_agents=bool(entry.get("may_spawn_agents")),
            may_integrate=bool(entry.get("may_integrate")),
            may_adjudicate=bool(entry.get("may_adjudicate")),
            counts_as_implementation=bool(entry.get("counts_as_implementation")),
        )
    if agent_type == "controller-direct":
        return AgentCapabilities(
            roles=("controller",),
            tools=("Read", "Grep", "Glob", "Bash", "Edit", "Write"),
            can_mutate=True,
            isolation="worktree",
            counts_as_implementation=True,
        )
    if agent_type in MUTATORS:
        role = agent_role(agent_type)
        return AgentCapabilities(
            roles=(role,),
            tools=("Read", "Grep", "Glob", "Bash", "Edit", "Write"),
            can_mutate=True,
            isolation="worktree",
            counts_as_implementation=True,
        )
    return AgentCapabilities(roles=(agent_role(agent_type),))


def agent_role(agent_type: str) -> str:
    """Resolve a native name's role, including registry-generated names."""
    if agent_type == "controller-direct":
        return "controller"
    entry = _registry_agent_manifest().get(agent_type)
    if entry is not None:
        return str(entry.get("role") or (entry.get("roles") or ["recon"])[0])
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
