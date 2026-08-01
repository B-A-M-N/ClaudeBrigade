"""Registry-backed identity for visible native specialists.

The generic role aliases remain stable compatibility names.  Model-qualified
specialists are declared beside the profile that owns them and are projected
into routing, scheduling, hooks, and the Claude Code launch manifest from one
source of truth.
"""

from __future__ import annotations

from typing import Any


GENERIC_ROLE_ALIASES: dict[str, str] = {
    "anthropic-brigade-recon": "recon",
    "anthropic-brigade-implementer": "implementer",
    "anthropic-brigade-adversary": "adversary",
    "anthropic-brigade-repairer": "repairer",
}


def specialist_manifest(
    registry: Any, profile_id: str | None = None
) -> dict[str, dict[str, str]]:
    """Return configured model-qualified native agents.

    A specialist without explicit launch identity remains a profile-level
    scheduling hint; it must not silently become a Claude Code Agent name.

    ``profile_id`` restricts the manifest to a single profile's specialists
    -- used when rendering a specific launch's agent directory, so an
    unrelated saved profile's specialist naming conflict can't block a
    launch that never selected it. ``None`` (the default) scans every saved
    profile, which remains required for process-wide authorization checks
    that have no single active profile to scope to.
    """
    result: dict[str, dict[str, str]] = {}
    if profile_id is not None:
        profiles = {profile_id: registry.profiles[profile_id]} if profile_id in registry.profiles else {}
    else:
        profiles = registry.profiles
    for profile_id, profile in sorted(profiles.items()):
        for specialist_id, specialist in sorted(profile.specialists.items()):
            native_name = specialist.native_agent_name
            public_alias = specialist.public_model_alias
            if not native_name and not public_alias:
                continue
            if not native_name or not public_alias:
                raise ValueError(
                    f"profile '{profile_id}' specialist '{specialist_id}' must "
                    "declare both native_agent_name and public_model_alias"
                )
            for role in specialist.roles:
                entry = {
                    "native_agent_name": native_name,
                    "public_model_alias": public_alias,
                    "model_id": specialist.model,
                    "role": role,
                    "profile_id": profile_id,
                    "specialist_id": specialist_id,
                }
                previous = result.get(native_name)
                if previous is not None and previous != entry:
                    raise ValueError(
                        f"native agent '{native_name}' has conflicting registry definitions"
                    )
                result[native_name] = entry
    return result


def role_model_aliases(registry: Any, profile_id: str | None = None) -> dict[str, str]:
    """Return stable role aliases plus configured specialist aliases."""
    result = dict(GENERIC_ROLE_ALIASES)
    for entry in specialist_manifest(registry, profile_id).values():
        alias = entry["public_model_alias"]
        role = entry["role"]
        previous = result.get(alias)
        if previous is not None and previous != role:
            raise ValueError(f"public model alias '{alias}' has conflicting roles")
        result[alias] = role
    return result


def role_model_bindings(registry: Any, profile_id: str | None = None) -> dict[str, str]:
    """Return explicit public/native specialist names to logical model IDs."""
    result: dict[str, str] = {}
    for entry in specialist_manifest(registry, profile_id).values():
        model_id = entry["model_id"]
        for alias in (entry["native_agent_name"], entry["public_model_alias"]):
            previous = result.get(alias)
            if previous is not None and previous != model_id:
                raise ValueError(f"agent identity '{alias}' has conflicting models")
            result[alias] = model_id
    return result


def native_agent_name(registry: Any, model_id: str, role: str) -> str:
    """Find a configured native name or fall back to the stable role agent."""
    candidates = sorted(
        entry["native_agent_name"]
        for entry in specialist_manifest(registry).values()
        if entry["model_id"] == model_id and entry["role"] == role
    )
    if candidates:
        return candidates[0]
    return f"brigade-{role}"
