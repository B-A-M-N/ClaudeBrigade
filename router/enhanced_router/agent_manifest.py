"""Registry-backed identities for every visible Claude Code worker.

The manifest is the single projection used by routing, scheduling, hooks, and
the launch ``--agents`` payload.  Native sidecar agents and profile
specialists share this projection; bounded coprocessors deliberately do not.
"""

from __future__ import annotations

from typing import Any

from enhanced_router.route_ladder import dedupe_candidates, target_candidates


GENERIC_ROLE_ALIASES: dict[str, str] = {
    "anthropic-brigade-recon": "recon",
    "anthropic-brigade-implementer": "implementer",
    "anthropic-brigade-adversary": "adversary",
    "anthropic-brigade-repairer": "repairer",
}

_ROLE_TEMPLATES = {
    "recon": "brigade-recon",
    "implementer": "brigade-implementer",
    "adversary": "brigade-adversary",
    "repairer": "brigade-repairer",
}

SLOT_ROLES = {
    "main": "controller",
    "sonnet": "implementer",
    "haiku": "recon",
    "opus": "adversary",
    "fable": "adversary",
    # These are policy lanes rather than durable role aliases.  The values
    # are used only for capability/certification validation; their public
    # slot entries intentionally expose no role identity.
    "background": "recon",
    "small-fast": "recon",
    "custom": "controller",
}

SLOT_ORDER = ("main", "sonnet", "haiku", "opus", "fable", "background")
SLOT_NATIVE_ALIASES = {
    "main": "main",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "opus": "opus",
    "fable": "fable",
}
SLOT_ENV_VARS = {
    "main": "ANTHROPIC_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "background": "ANTHROPIC_SMALL_FAST_MODEL",
    "custom": "ANTHROPIC_CUSTOM_MODEL_OPTION",
}


def _slot_entries(registry: Any, profile_id: str | None) -> dict[str, dict[str, Any]]:
    """Project a selected profile into stable Claude Code model slots.

    Legacy profiles get a deterministic compatibility projection from their
    controller/role routes. Explicit ``slots`` remain authoritative when a
    profile declares them. Sidecar agents are intentionally not included in
    this mapping: they are additional named workers with their own model IDs.
    """
    if profile_id is None or profile_id not in registry.profiles:
        return {}
    profile = registry.profiles[profile_id]
    role_targets = {
        "main": profile.controller_route() or profile.route_target("implementer"),
        "sonnet": profile.route_target("implementer"),
        "haiku": profile.route_target("recon"),
        "opus": profile.route_target("adversary"),
        "fable": profile.route_target("adversary"),
        "background": profile.route_target("recon"),
    }
    entries: dict[str, dict[str, Any]] = {}
    for slot in SLOT_ORDER:
        role_target = role_targets[slot]
        # ``small-fast`` is the fi-flow spelling.  ``background`` is the
        # ClaudeBrigade spelling exposed in the configuration UI and docs.
        slot_route = profile.slots.get(slot)
        if slot == "background" and slot_route is None:
            slot_route = profile.slots.get("small-fast")
        if slot == "background" and slot_route is None and not profile.slots:
            continue
        candidates = (
            target_candidates(slot_route)
            if slot_route is not None
            else target_candidates(role_target)
        )
        if slot == "main" and slot_route is not None:
            # Keep the legacy/profile-level controller ladder visible when a
            # profile declares the Main slot's primary separately.  The slot
            # primary remains first; exact-identity deduplication prevents a
            # duplicated controller primary from changing order.
            candidates = [
                *candidates,
                *target_candidates(profile.controller_route()),
            ]
        candidates = dedupe_candidates(candidates)
        primary = candidates[0]
        model_id = primary["model"]
        endpoint = primary.get("endpoint", "auto")
        provider_id = primary.get("provider_id")
        public_model_alias = f"anthropic-brigade-slot-{slot}"
        entries[slot] = {
            "slot": slot,
            # Only the five Claude-native names may be emitted as native
            # aliases.  Background and custom are concrete router lanes.
            "model_alias": SLOT_NATIVE_ALIASES.get(slot, public_model_alias),
            "native_alias": SLOT_NATIVE_ALIASES.get(slot),
            "public_model_alias": public_model_alias,
            "model_id": model_id,
            "endpoint": endpoint,
            "provider_id": provider_id,
            "fallbacks": candidates[1:],
            "reserve_class": slot_route.reserve_class if slot_route else (
                "controller" if slot == "main" else "critical" if slot == "fable" else "worker"
            ),
            "role": SLOT_ROLES[slot] if slot in SLOT_NATIVE_ALIASES else None,
            "roles": [SLOT_ROLES[slot]] if slot in SLOT_NATIVE_ALIASES else [],
            "environment_variable": SLOT_ENV_VARS[slot],
            "is_native_alias": slot in SLOT_NATIVE_ALIASES,
            "is_background_lane": slot == "background",
            "is_custom_lane": slot == "custom",
            "source_kind": "model_slot",
            "source_id": slot,
        }
    custom_route = profile.slots.get("custom")
    if custom_route is not None:
        candidates = dedupe_candidates(target_candidates(custom_route))
        primary = candidates[0]
        entries["custom"] = {
            "slot": "custom",
            "model_alias": "anthropic-brigade-slot-custom",
            "native_alias": None,
            "public_model_alias": "anthropic-brigade-slot-custom",
            "model_id": primary["model"],
            "endpoint": primary.get("endpoint", "auto"),
            "provider_id": primary.get("provider_id"),
            "fallbacks": candidates[1:],
            "reserve_class": custom_route.reserve_class,
            "role": None,
            "roles": [],
            "environment_variable": SLOT_ENV_VARS["custom"],
            "is_native_alias": False,
            "is_background_lane": False,
            "is_custom_lane": True,
            "source_kind": "model_slot",
            "source_id": "custom",
        }
    return entries


def slot_alias_manifest(registry: Any, profile_id: str | None = None) -> dict[str, dict[str, Any]]:
    """Return the selected profile's native alias projection."""
    return _slot_entries(registry, profile_id)


def _profile_agent_entries(registry: Any, profile_id: str | None) -> dict[str, dict[str, Any]]:
    if profile_id is None or profile_id not in registry.profiles:
        return {}
    profile = registry.profiles[profile_id]
    slots = _slot_entries(registry, profile_id)
    result: dict[str, dict[str, Any]] = {}
    for agent_id, spec in sorted(profile.agents.items()):
        native_name = agent_id if agent_id.startswith("brigade-") else f"brigade-{agent_id}"
        slot = spec.slot
        slot_entry = slots.get(slot) if slot else None
        model_id = spec.explicit_model or (slot_entry or {}).get("model_id")
        if not model_id:
            raise ValueError(f"profile '{profile_id}' agent '{agent_id}' has no model route")
        primary_role = (spec.roles or [SLOT_ROLES.get(slot or "haiku", "recon")])[0]
        model_alias = (
            (slot_entry or {}).get("model_alias")
            or spec.explicit_model
        )
        result[native_name] = {
            "native_agent_name": native_name,
            "public_model_alias": (
                (slot_entry or {}).get("public_model_alias")
                or f"anthropic-brigade-{agent_id}"
            ),
            "model_alias": model_alias,
            "model_id": model_id,
            "roles": list(spec.roles),
            "role": primary_role,
            "source_kind": "profile_agent",
            "source_id": agent_id,
            "profile_id": profile_id,
            "agent_id": agent_id,
            "template": spec.template,
            "can_mutate": spec.can_mutate,
            "counts_as_implementation": spec.counts_as_implementation,
            "tools": list(spec.tools),
            "disallowed_tools": list(spec.disallowed_tools),
            "isolation": spec.isolation,
            "background": spec.background,
            "max_turns": spec.max_turns,
            "effort": spec.effort,
            "permission_mode": spec.permission_mode,
            "may_spawn_agents": spec.may_spawn_agents,
            "may_integrate": spec.may_integrate,
            "may_adjudicate": spec.may_adjudicate,
            "enabled": True,
        }
    return result


def _stable_entries(registry: Any, profile_id: str | None) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    profile = registry.profiles.get(profile_id) if profile_id else None
    slots = _slot_entries(registry, profile_id)
    role_slots = {
        "implementer": "sonnet",
        "recon": "haiku",
        "adversary": "opus",
        "repairer": "opus",
    }
    for role, native_name in _ROLE_TEMPLATES.items():
        model_id = ""
        if profile is not None:
            model_id = profile.route_target(role).model
        slot = role_slots.get(role)
        slot_entry = slots.get(slot) if slot else None
        entries[native_name] = {
            "native_agent_name": native_name,
            "public_model_alias": (slot_entry or {}).get("public_model_alias", f"anthropic-brigade-{role}"),
            "model_alias": (slot_entry or {}).get("model_alias"),
            # A slot is a route projection of the role. Keep the durable role
            # model as the fallback for old profiles and test doubles.
            "model_id": (slot_entry or {}).get("model_id") or model_id,
            "role_model_id": model_id,
            "slot": slot,
            "roles": [role],
            "role": role,
            "source_kind": "stable_role",
            "source_id": role,
            "can_mutate": role in {"implementer", "repairer"},
            "counts_as_implementation": role in {"implementer", "repairer"},
            "tools": ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]
            if role in {"implementer", "repairer"}
            else ["Read", "Grep", "Glob", "Bash"],
            "disallowed_tools": [],
            "isolation": "worktree" if role in {"implementer", "repairer"} else "none",
            "background": role != "implementer" and role != "repairer",
            "max_turns": 140 if role == "implementer" else 100,
            "effort": "high",
            "permission_mode": "acceptEdits" if role in {"implementer", "repairer"} else "plan",
            "template": native_name,
            "may_spawn_agents": False,
            "may_integrate": False,
            "may_adjudicate": role == "adversary",
            "enabled": True,
        }
    return entries


def _specialist_entries(registry: Any, profile_id: str | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if profile_id is not None:
        profiles = {
            profile_id: registry.profiles[profile_id]
        } if profile_id in registry.profiles else {}
    else:
        profiles = registry.profiles
    for selected_profile_id, profile in sorted(profiles.items()):
        for specialist_id, specialist in sorted(profile.specialists.items()):
            # Catalog-only specialists are intentionally not launchable.  Do
            # not infer this from missing identity fields: a malformed
            # launchable specialist must fail closed during manifest
            # generation instead of silently disappearing.
            if not specialist.launchable:
                continue
            native_name = specialist.native_agent_name
            public_alias = specialist.public_model_alias
            if not native_name or not public_alias:
                raise ValueError(
                    f"profile '{selected_profile_id}' specialist '{specialist_id}' must "
                    "declare both native_agent_name and public_model_alias"
                )
            roles = list(specialist.roles)
            primary_role = roles[0] if roles else "recon"
            entry = {
                "native_agent_name": native_name,
                "public_model_alias": public_alias,
                "model_id": specialist.model,
                "roles": roles,
                "role": primary_role,
                "source_kind": "inference_specialist",
                "source_id": specialist_id,
                "profile_id": selected_profile_id,
                "specialist_id": specialist_id,
                "can_mutate": primary_role in {"implementer", "repairer"},
                "counts_as_implementation": primary_role in {"implementer", "repairer"},
                "tools": [],
                "disallowed_tools": [],
                "isolation": "worktree" if primary_role in {"implementer", "repairer"} else "none",
                "background": primary_role not in {"implementer", "repairer"},
                "max_turns": 100,
                "effort": "high" if primary_role in {"implementer", "repairer"} else "medium",
                "permission_mode": "acceptEdits" if primary_role in {"implementer", "repairer"} else "plan",
                "template": _ROLE_TEMPLATES.get(primary_role, "brigade-recon"),
                "may_spawn_agents": False,
                "may_integrate": False,
                "may_adjudicate": primary_role == "adversary",
                "enabled": True,
            }
            previous = result.get(native_name)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"native agent '{native_name}' has conflicting registry definitions"
                )
            result[native_name] = entry
    return result


def _sidecar_entries(
    registry: Any, sidecar_profile_id: str | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source_id, sidecar in sorted(registry.resolve_sidecar_agents(sidecar_profile_id).items()):
        roles = list(sidecar.roles)
        primary_role = roles[0] if roles else "recon"
        result[sidecar.native_agent_name] = {
            "native_agent_name": sidecar.native_agent_name,
            "worker_id": sidecar.worker_id or source_id,
            "public_model_alias": sidecar.public_model_alias,
            "model_alias": sidecar.public_model_alias,
            "model_id": sidecar.model_id,
            "roles": roles,
            "role": primary_role,
            "source_kind": "sidecar_agent",
            "source_id": source_id,
            "sidecar_agent_id": source_id,
            "description": sidecar.description,
            "system_prompt": sidecar.system_prompt,
            "can_mutate": sidecar.can_mutate,
            "counts_as_implementation": sidecar.counts_as_implementation,
            "tools": list(sidecar.tools),
            "disallowed_tools": list(sidecar.disallowed_tools),
            "isolation": sidecar.isolation,
            "background": sidecar.background,
            "max_turns": sidecar.max_turns,
            "effort": sidecar.effort,
            "permission_mode": sidecar.permission_mode,
            "template": sidecar.template or _ROLE_TEMPLATES.get(primary_role, "brigade-recon"),
            "endpoint": sidecar.endpoint,
            "provider_id": sidecar.provider_id,
            "fallback_models": list(sidecar.fallback_models),
            "fallback_routes": [
                candidate.model_dump(exclude_none=True)
                for candidate in sidecar.fallback_routes
            ],
            "max_parallelism": sidecar.max_parallelism,
            "may_spawn_agents": sidecar.may_spawn_agents,
            "may_integrate": sidecar.may_integrate,
            "may_adjudicate": sidecar.may_adjudicate,
            "enabled": sidecar.enabled,
        }
    return result


def native_worker_manifest(
    registry: Any,
    inference_profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge stable roles, profile specialists, and selected sidecar agents."""
    result = _stable_entries(registry, inference_profile_id)
    for lane in (_profile_agent_entries(registry, inference_profile_id),
                 _specialist_entries(registry, inference_profile_id),
                 _sidecar_entries(registry, sidecar_profile_id)):
        for native_name, entry in lane.items():
            previous = result.get(native_name)
            if previous is not None and entry.get("source_kind") == "profile_agent":
                # A profile agent may deliberately refine a stable role
                # identity (tools, prompt template, native slot) without
                # creating a second identity for the same role.
                merged = dict(previous)
                merged.update(entry)
                result[native_name] = merged
            elif previous is not None and previous != entry:
                raise ValueError(
                    f"native agent '{native_name}' has conflicting registry definitions"
                )
            else:
                result[native_name] = entry

    aliases: dict[str, str] = {}
    for entry in result.values():
        alias = str(entry["public_model_alias"])
        model_id = str(entry.get("model_id") or "")
        previous = aliases.get(alias)
        if previous is not None and previous != model_id:
            raise ValueError(f"public model alias '{alias}' has conflicting models")
        aliases[alias] = model_id
    return result


def specialist_manifest(
    registry: Any, profile_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compatibility name for the complete native worker manifest."""
    return native_worker_manifest(registry, profile_id, None)


def role_model_aliases(
    registry: Any,
    profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> dict[str, str]:
    """Return stable role aliases plus specialist and sidecar aliases."""
    result = dict(GENERIC_ROLE_ALIASES)
    for slot, entry in slot_alias_manifest(registry, profile_id).items():
        # Background and custom are model lanes, not durable role aliases.
        # They are resolved by the slot projection itself and must not acquire
        # a fake role that would trigger native-agent identity requirements.
        if entry.get("role") is not None:
            result[entry["public_model_alias"]] = entry["role"]
    for entry in native_worker_manifest(registry, profile_id, sidecar_profile_id).values():
        alias = str(entry["public_model_alias"])
        roles = list(entry.get("roles") or [entry.get("role", "recon")])
        role = str(roles[0])
        previous = result.get(alias)
        if previous is not None and previous != role:
            raise ValueError(f"public model alias '{alias}' has conflicting roles")
        result[alias] = role
    return result


def role_model_bindings(
    registry: Any,
    profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> dict[str, str]:
    """Return public/native identities to logical model IDs."""
    result: dict[str, str] = {}
    for entry in slot_alias_manifest(registry, profile_id).values():
        result[entry["public_model_alias"]] = str(entry["model_id"])
        result[entry["model_alias"]] = str(entry["model_id"])
    for entry in native_worker_manifest(registry, profile_id, sidecar_profile_id).values():
        model_id = str(entry.get("model_id") or "")
        if not model_id:
            continue
        for alias in (entry["native_agent_name"], entry["public_model_alias"]):
            previous = result.get(alias)
            if previous is not None and previous != model_id:
                raise ValueError(f"agent identity '{alias}' has conflicting models")
            result[alias] = model_id
    return result


def resolve_native_worker(
    registry: Any,
    worker_id: str | None,
    model_id: str,
    role: str,
    *,
    inference_profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> dict[str, Any]:
    """Resolve an explicit worker identity before falling back to model/role."""
    manifest = native_worker_manifest(registry, inference_profile_id, sidecar_profile_id)
    if worker_id:
        for entry in manifest.values():
            if worker_id in {
                entry["native_agent_name"], entry.get("source_id"),
                entry.get("sidecar_agent_id"), entry.get("specialist_id"),
                entry.get("worker_id"),
            }:
                return entry
        raise KeyError(f"Unknown native worker: {worker_id}")
    candidates = [
        entry for entry in manifest.values()
        if entry.get("model_id") == model_id
        and role in set(entry.get("roles") or [entry.get("role")])
    ]
    if candidates:
        return sorted(candidates, key=lambda item: item["native_agent_name"])[0]
    return manifest.get(_ROLE_TEMPLATES.get(role, f"brigade-{role}"), {
        "native_agent_name": f"brigade-{role}",
        "public_model_alias": f"anthropic-brigade-{role}",
        "model_id": model_id,
        "roles": [role],
        "role": role,
        "source_kind": "stable_role",
        "source_id": role,
        "can_mutate": role in {"implementer", "repairer"},
        "counts_as_implementation": role in {"implementer", "repairer"},
        "isolation": "worktree" if role in {"implementer", "repairer"} else "none",
    })


def native_agent_name(
    registry: Any, model_id: str, role: str, worker_id: str | None = None,
) -> str:
    """Return an explicit worker name or the stable model-neutral role name.

    A model/role lookup is intentionally *not* allowed to select a
    model-qualified specialist.  That would make backing-model identity part
    of workflow authority again. Callers that genuinely need a specialist
    must provide its explicit semantic ``worker_id``.
    """
    if worker_id:
        return str(resolve_native_worker(registry, worker_id, model_id, role)["native_agent_name"])
    return f"brigade-{role}"
