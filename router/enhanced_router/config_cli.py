"""Interactive saved configuration manager for ClaudeBrigade.

The CLI deliberately displays credential names and availability only. Secret
values are loaded from the OS keyring when available, with the same locked
providers.env parser as a compatibility fallback. They are never echoed or
written to model/sidecar YAML.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from enhanced_router.bootstrap_env import (
    ALLOWED_PROVIDER_KEYS,
    PROVIDER_SECRET_KEYS,
    load_router_credentials,
)
from enhanced_router.credential_store import (
    CredentialStoreUnavailable,
    available as credential_store_available,
    backend_label,
    delete_slot,
    read_slots,
    set_slot,
    slot_names,
)
from enhanced_router.env_parser import parse_env_file
from enhanced_router.registry import ModelRegistry

_ROLES = ("recon", "implementer", "adversary", "repairer")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _config_dir(value: str | None) -> Path:
    return Path(
        value
        or os.environ.get("BRIGADE_CONFIG_DIR")
        or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "claude-brigade"
    ).expanduser()


def _load_registry(config_dir: Path) -> tuple[ModelRegistry, tuple[str, ...]]:
    config_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_router_credentials(config_dir)
    # The values stay in this short-lived CLI process only. They are needed to
    # determine which configured transports are usable; they are never printed.
    os.environ.update(loaded.provider_env)
    registry = ModelRegistry(config_dir)
    registry.load_models()
    # Preserve older user-owned catalogs while making newly installed
    # sidecars/fastpath definitions resolvable.  The installer stores the
    # current bundled catalog as models.yaml.example on upgrades; source
    # checkouts use the repository config directory instead.  Only the
    # router-owned compatibility models are overlaid, and user definitions
    # always remain authoritative.
    bundled_models = config_dir / "models.yaml.example"
    if not bundled_models.exists():
        source_models = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
        if source_models.exists():
            bundled_models = source_models
    if bundled_models.exists() and bundled_models.resolve() != (config_dir / "models.yaml").resolve():
        bundled = ModelRegistry(bundled_models.parent)
        bundled.load_models(bundled_models)
        for model_id, spec in bundled.models.items():
            if spec.backend == "anthropic-passthrough" or spec.provider_id == "freeinference":
                registry.models.setdefault(model_id, spec)
    registry.load_profiles()
    registry.load_workflows()
    registry.load_providers()
    registry.load_fastpath()
    registry.load_sidecars()
    registry._validate_cross_refs()
    return registry, loaded.provider_keys


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked configuration: {path}")
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent), text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_provider_env(path: Path, values: dict[str, str]) -> None:
    """Atomically write owner-readable provider credentials without echoing them."""
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked credential file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent), text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "# ClaudeBrigade provider credentials; managed by claude-brigade-config.\n"
                + "".join(
                    f"{key}={json.dumps(values[key])}\n"
                    for key in sorted(values)
                    if values[key]
                )
            )
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(label: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Enter a number from {minimum} to {maximum}.")


def _prompt_float(label: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = _prompt(label, str(default))
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Enter a number from {minimum} to {maximum}.")


def _choose(label: str, options: list[tuple[str, str]], default: int = 1) -> str:
    if not options:
        raise RuntimeError(f"No choices are available for {label}.")
    print(f"\n{label}")
    for number, (value, description) in enumerate(options, start=1):
        print(f"  {number}. {value} — {description}")
    while True:
        raw = _prompt("Choose", str(default))
        try:
            index = int(raw)
        except ValueError:
            print("Choose one of the listed numbers.")
            continue
        if 1 <= index <= len(options):
            return options[index - 1][0]
        print("Choose one of the listed numbers.")


def _choose_model(
    label: str, options: list[tuple[str, str]], default_id: str = ""
) -> str:
    """Search a potentially large discovered catalog before choosing."""
    while True:
        query = input(f"Search {label} (blank lists all): ").strip().lower()
        filtered = options if not query else [
            item for item in options
            if query in item[0].lower() or query in item[1].lower()
        ]
        if not filtered:
            print(f"No matches for '{query}'. Press Enter with nothing typed to list all {len(options)} options.")
            continue
        default = next(
            (index for index, item in enumerate(filtered, 1) if item[0] == default_id),
            1,
        )
        return _choose(label, filtered, default)


def _model_provider(registry: ModelRegistry, model_id: str, endpoint: str = "auto") -> tuple[str, str, str, bool]:
    model = registry.get_model(model_id)
    candidates = []
    if endpoint != "auto" and endpoint in model.endpoints:
        candidates = [model.endpoints[endpoint]]
    else:
        candidates = list(model.endpoints.values()) or [model]
    for candidate in candidates:
        provider_id = getattr(candidate, "provider_id", None) or model.provider_id or "local"
        provider = registry.providers.get(provider_id)
        key_env = getattr(candidate, "api_key_env", None) or model.api_key_env
        if provider is not None:
            key_env = key_env or provider.api_key_env
        configured = not key_env or bool(os.environ.get(key_env))
        backend = getattr(candidate, "backend", model.backend)
        if backend == "direct-anthropic":
            configured = configured and bool(
                getattr(candidate, "api_base", None)
                or model.api_base
                or getattr(candidate, "api_base_env", None)
                or model.api_base_env
            )
        if backend == "litellm":
            configured = configured and bool(
                getattr(candidate, "litellm_model", None) or model.litellm_model
            )
        if configured:
            return provider_id, backend, key_env or "no key required", True
    provider_id = model.provider_id or "local"
    return provider_id, model.backend, model.api_key_env or "no key required", False


def _model_choices(
    registry: ModelRegistry,
    *,
    role: str | None = None,
    controller: bool = False,
) -> list[tuple[str, str]]:
    """List selectable models for a role (or the controller).

    A freshly-discovered model (see registry.apply_discovered_catalog) starts
    with an empty ``allowed_roles`` set, and ``capabilities.controller_eligible
    = False``, by design -- discovery never grants roles or controller
    eligibility on its own. Without special-casing that here, such a model
    could never be searched or selected for ANY role or as the controller:
    the filters below would always exclude it, and nothing else ever grants
    it. Treat "not yet decided" as offerable; picking it is what grants the
    role or controller eligibility (see configure_inference /
    _grant_controller_eligible). Once a model has been explicitly restricted
    to a non-empty set of roles, that restriction is still enforced normally.
    """
    choices: list[tuple[str, str]] = []
    for model_id, model in sorted(registry.models.items(), key=lambda item: item[0]):
        if not model.enabled or model.backend == "anthropic-passthrough" and not controller:
            continue
        ungranted = False
        if controller:
            if not model.capabilities.controller_eligible:
                ungranted = True
        elif role is not None:
            if model.allowed_roles:
                if role not in model.allowed_roles:
                    continue
            else:
                ungranted = True
        provider, backend, key_env, configured = _model_provider(registry, model_id)
        if not configured:
            continue
        label = f"{model.display_name}; {backend}; provider={provider}; key={key_env}"
        if ungranted:
            suffix = "controller eligibility" if controller else "the role"
            label += f"; UNCERTIFIED -- selecting this grants it {suffix}"
        choices.append((model_id, label))
    return choices


def _grant_discovered_role(config_dir: Path, registry: ModelRegistry, model_id: str, role: str) -> None:
    """Grant *role* to a discovered model that has no roles assigned yet.

    Persisted into discovered_models.yaml (the same file
    Registry._persist_discovered_catalog writes) so the grant survives a
    future `refresh` re-running discovery -- that function preserves any
    existing allowed_roles/capabilities it finds there rather than
    overwriting them. Also updates the in-memory registry so the rest of
    the current wizard session sees the grant immediately.
    """
    path = config_dir / "discovered_models.yaml"
    raw = _read_yaml(path)
    models = raw.setdefault("models", {})
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        return
    roles = list(entry.get("allowed_roles") or [])
    if role not in roles:
        roles.append(role)
    entry["allowed_roles"] = roles
    if role in {"implementer", "repairer"}:
        caps = dict(entry.get("capabilities") or {})
        caps["mutation"] = True
        entry["capabilities"] = caps
    _write_yaml(path, raw)

    spec = registry.models.get(model_id)
    if spec is not None:
        updates: dict[str, Any] = {"allowed_roles": spec.allowed_roles | {role}}
        if role in {"implementer", "repairer"}:
            updates["capabilities"] = spec.capabilities.model_copy(update={"mutation": True})
        registry.models[model_id] = spec.model_copy(update=updates)


def _grant_controller_eligible(config_dir: Path, registry: ModelRegistry, model_id: str) -> None:
    """Grant controller eligibility to a model that isn't certified for it.

    Same persistence pattern as _grant_discovered_role: written into
    discovered_models.yaml so it survives a future `refresh`, and reflected
    in-memory immediately. The launcher (bin/claude-brigade) independently
    re-checks `capabilities.controller_eligible or backend ==
    "anthropic-passthrough"` before it will actually launch with this model
    as the controller, so this grant is what makes that check pass -- without
    it, a model picked here would be rejected at launch time with "is not
    controller-compatible" even though the wizard let you select it.
    """
    path = config_dir / "discovered_models.yaml"
    raw = _read_yaml(path)
    models = raw.setdefault("models", {})
    entry = models.get(model_id)
    if isinstance(entry, dict):
        caps = dict(entry.get("capabilities") or {})
        caps["controller_eligible"] = True
        entry["capabilities"] = caps
        _write_yaml(path, raw)

    spec = registry.models.get(model_id)
    if spec is not None:
        registry.models[model_id] = spec.model_copy(
            update={"capabilities": spec.capabilities.model_copy(update={"controller_eligible": True})}
        )


def _target_label(role: str | None, controller: bool) -> str:
    return "controller" if controller else str(role)


def _record_certification(config_dir: Path, model_id: str, target: str, probe_record: dict[str, Any]) -> None:
    """Persist probe evidence to model_certifications.yaml.

    This is an evidence record, not an authority grant by itself -- the
    caller only reaches the actual allowed_roles/controller_eligible grant
    when probe_record['status'] == 'certified'.
    """
    path = config_dir / "model_certifications.yaml"
    raw = _read_yaml(path)
    certifications = raw.setdefault("certifications", {})
    per_model = certifications.setdefault(model_id, {})
    per_model[target] = probe_record
    _write_yaml(path, raw)


def _record_override(config_dir: Path, model_id: str, target: str, reason: str) -> None:
    """Persist an explicit, unverified operator decision to model_overrides.yaml.

    Distinct from model_certifications.yaml on purpose: this is a decision
    the operator made without passing evidence, not a claim that the model
    was tested and works.
    """
    path = config_dir / "model_overrides.yaml"
    raw = _read_yaml(path)
    overrides = raw.setdefault("overrides", {})
    per_model = overrides.setdefault(model_id, {})
    per_model[target] = {
        "reason": reason,
        "granted_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    _write_yaml(path, raw)


def _apply_grant(config_dir: Path, registry: ModelRegistry, model_id: str, *, role: str | None, controller: bool) -> None:
    if controller:
        _grant_controller_eligible(config_dir, registry, model_id)
    elif role is not None:
        _grant_discovered_role(config_dir, registry, model_id, role)


def _confirm_and_grant(
    config_dir: Path, registry: ModelRegistry, model_id: str, *, role: str | None = None, controller: bool = False,
) -> bool:
    """Gate an uncertified model behind an explicit test-or-override choice.

    Returns True if the model is now usable for this target (already was,
    just got certified, or was explicitly overridden). Returns False if the
    operator backed out, so the caller should let them pick a different
    model instead of silently proceeding.

    Selecting a model from the list must never by itself grant it anything
    -- that was the bug in the original _grant_* wiring, which treated
    "the user typed this model's number" as equivalent to "this model is
    certified for this role."
    """
    target = _target_label(role, controller)
    spec = registry.get_model(model_id)
    already_ok = spec.capabilities.controller_eligible if controller else (role in spec.allowed_roles if role else True)
    if already_ok:
        return True

    print(
        f"\n'{model_id}' is unverified for {target}. Discovery never grants roles or "
        "controller eligibility on its own -- pick how to proceed:"
    )
    print("  t) Run a compatibility test now (one real API call, uses your credentials)")
    print("  o) Override without testing (explicit, at your own risk)")
    print("  <anything else>) Cancel and pick a different model")
    choice = input("Choice: ").strip().lower()

    if choice == "t":
        from enhanced_router.model_probe import probe_model

        print(f"Probing '{model_id}' for {target} compatibility...")
        result = probe_model(model_id, spec, config_dir=str(config_dir))
        _record_certification(config_dir, model_id, target, result.to_record())
        if result.passed:
            print(f"Passed: {model_id} demonstrated tool-call support for {target}.")
            _apply_grant(config_dir, registry, model_id, role=role, controller=controller)
            return True
        print(f"Failed: {result.error or 'no tool call observed'}.")
        if input("Override and use it anyway despite the failed test? [y/N]: ").strip().lower() != "y":
            return False
        choice = "o"

    if choice == "o":
        reason = _prompt("Reason for overriding without certification", "operator decision")
        confirm = input(
            f"Type OVERRIDE to confirm using an unverified model for {target} "
            "(it may fail unpredictably, including mid-mutation): "
        ).strip()
        if confirm != "OVERRIDE":
            print("Not confirmed; cancelled.")
            return False
        _record_override(config_dir, model_id, target, reason)
        _apply_grant(config_dir, registry, model_id, role=role, controller=controller)
        return True

    return False


def _profile_model(profile: dict[str, Any], role: str) -> tuple[str, str, list[str]]:
    raw = profile.get(role, "")
    if isinstance(raw, dict):
        return (
            str(raw.get("model", "")),
            str(raw.get("endpoint", "auto")),
            [str(item) for item in raw.get("fallback_models", []) if isinstance(item, str)],
        )
    return str(raw), "auto", []


def configure_inference(config_dir: Path, registry: ModelRegistry) -> str:
    raw = _read_yaml(config_dir / "profiles.yaml")
    profiles = raw.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        raw["profiles"] = profiles
    existing = sorted(str(item) for item in profiles)
    print(
        "\nA profile is a named, reusable set of controller + role model "
        "assignments -- the actual model choices come next, right after this."
    )
    if existing:
        print(f"Existing profiles: {', '.join(existing)}")
        print("Press Enter to edit one of those, or type a new name to create a separate saved setup.")
    else:
        print("No profiles saved yet -- name this one (letters, numbers, '.', '_', '-').")
    profile_id = _prompt("Profile name", existing[0] if existing else "my-profile")
    if not _NAME_RE.fullmatch(profile_id):
        raise ValueError("profile IDs may contain letters, numbers, '.', '_' and '-'")
    current = dict(profiles.get(profile_id) or {})
    print("\n=== Main models: controller + recon/implementer/adversary/repairer ===")
    print("For each, type part of a name to search, or press Enter to list everything.")
    controller_default = str(current.get("controller_model") or "")
    controller_choices = _model_choices(registry, controller=True)
    if controller_choices:
        while True:
            controller_model = _choose_model(
                "Controller model", controller_choices,
                controller_default,
            )
            if _confirm_and_grant(config_dir, registry, controller_model, controller=True):
                break
        current["controller_model"] = controller_model
    for role in _ROLES:
        choices = _model_choices(registry, role=role)
        if not choices:
            raise RuntimeError(f"No credential-backed model is available for role '{role}'.")
        previous, previous_endpoint, previous_fallbacks = _profile_model(current, role)
        while True:
            selected = _choose_model(
                f"{role} model", choices,
                previous,
            )
            if _confirm_and_grant(config_dir, registry, selected, role=role):
                break
        model = registry.get_model(selected)
        endpoints = [("auto", "registry endpoint selection")]
        endpoints.extend((endpoint_id, f"configured {endpoint.backend} endpoint") for endpoint_id, endpoint in sorted(model.endpoints.items()))
        endpoint = _choose(
            f"{role} endpoint for {selected}", endpoints,
            next((i for i, item in enumerate(endpoints, 1) if item[0] == previous_endpoint), 1),
        )
        fallback_options = [item for item in choices if item[0] != selected]
        fallback_raw = _prompt(
            f"{role} fallback model IDs (comma-separated, blank for none)",
            ",".join(previous_fallbacks),
        )
        fallback_models = [item.strip() for item in fallback_raw.split(",") if item.strip()]
        allowed_fallbacks = {item[0] for item in fallback_options}
        unknown_fallbacks = [item for item in fallback_models if item not in allowed_fallbacks]
        if unknown_fallbacks:
            raise ValueError(
                f"fallback model(s) unavailable for role '{role}': {', '.join(unknown_fallbacks)}"
            )
        confirmed_fallbacks = []
        for fallback_id in fallback_models:
            if _confirm_and_grant(config_dir, registry, fallback_id, role=role):
                confirmed_fallbacks.append(fallback_id)
            else:
                print(f"Dropping '{fallback_id}' from {role} fallbacks (not confirmed).")
        fallback_models = confirmed_fallbacks
        if endpoint == "auto" and not fallback_models:
            current[role] = selected
        else:
            current[role] = {
                "model": selected,
                "endpoint": endpoint,
                "fallback_models": fallback_models,
            }
    profiles[profile_id] = current
    _write_yaml(config_dir / "profiles.yaml", raw)
    print(f"Saved inference profile '{profile_id}' to {config_dir / 'profiles.yaml'}.")
    print(f"Use it with: claude-brigade --brigade-profile {profile_id}")
    return profile_id


def configure_fastpath(config_dir: Path, registry: ModelRegistry) -> None:
    """Configure the always-on fastpath coprocessor (fastpath.yaml).

    This is distinct from sidecars.yaml: the fastpath model runs
    automatically on every request via /internal/fastpath/route and
    /internal/fastpath/verify. A sidecars.yaml entry, by contrast, does
    nothing until a workflow phase explicitly opts in with
    `execution_kind: sidecar_call` and `sidecar: <id>`.
    """
    print("\n=== Sidecar & fastpath models: coprocessor ===")
    raw = _read_yaml(config_dir / "fastpath.yaml")
    fastpath = raw.setdefault("fastpath", {})
    if not isinstance(fastpath, dict):
        fastpath = {}
        raw["fastpath"] = fastpath
    # registry._validate_cross_refs() rejects a write-tool-certified fastpath
    # model outright ("fastpath model cannot be write-tool certified") -- but
    # only at the NEXT registry load. Without filtering here, the wizard
    # would happily save a choice that crashes the very next launch, the
    # same failure mode this whole session has been chasing.
    choices = [
        item for item in _model_choices(registry)
        if not registry.get_model(item[0]).capabilities.write_tool_certified
    ]
    if not choices:
        raise RuntimeError("No enabled credential-backed model is available for the fastpath coprocessor.")
    previous_model = str(fastpath.get("model_id") or "")
    model_id = _choose_model("Fastpath coprocessor model", choices, previous_model)
    model = registry.get_model(model_id)
    endpoints = [("auto", "registry endpoint selection")]
    endpoints.extend((endpoint_id, f"configured {endpoint.backend} endpoint") for endpoint_id, endpoint in sorted(model.endpoints.items()))
    endpoint = _choose(
        f"Fastpath endpoint for {model_id}", endpoints,
        next((i for i, item in enumerate(endpoints, 1) if item[0] == fastpath.get("endpoint", "auto")), 1),
    )
    fastpath["model_id"] = model_id
    fastpath["endpoint"] = endpoint
    fastpath.setdefault("enabled", True)
    fastpath.setdefault("modes", ["route", "verify"])
    # FastpathConfigSpec.timeout_seconds caps at 30 (gt=0, le=30) -- the
    # schema is the source of truth, not a second hardcoded bound here.
    fastpath["timeout_seconds"] = _prompt_float(
        "Fastpath timeout seconds", float(fastpath.get("timeout_seconds", 5)), 1, 30,
    )
    _write_yaml(config_dir / "fastpath.yaml", raw)
    print(f"Saved fastpath coprocessor config to {config_dir / 'fastpath.yaml'}.")


def configure_sidecar(config_dir: Path, registry: ModelRegistry) -> None:
    print("\n=== Sidecar & fastpath models: named sidecar (dormant until a workflow phase references it) ===")
    raw = _read_yaml(config_dir / "sidecars.yaml")
    sidecars = raw.setdefault("sidecars", {})
    if not isinstance(sidecars, dict):
        sidecars = {}
        raw["sidecars"] = sidecars
    existing = sorted(str(item) for item in sidecars)
    sidecar_id = _prompt("Saved sidecar ID", existing[0] if existing else "verification_reviewer")
    if not _NAME_RE.fullmatch(sidecar_id):
        raise ValueError("sidecar IDs may contain letters, numbers, '.', '_' and '-'")
    current = dict(sidecars.get(sidecar_id) or {})
    choices = _model_choices(registry)
    if not choices:
        raise RuntimeError("No enabled credential-backed sidecar model is available.")
    previous_model = str(current.get("model_id") or "")
    model_id = _choose_model(
        "Sidecar inference model", choices,
        previous_model,
    )
    model = registry.get_model(model_id)
    endpoints = [("auto", "registry endpoint selection")]
    endpoints.extend((endpoint_id, f"configured {endpoint.backend} endpoint") for endpoint_id, endpoint in sorted(model.endpoints.items()))
    endpoint = _choose(
        f"Sidecar endpoint for {model_id}", endpoints,
        next((i for i, item in enumerate(endpoints, 1) if item[0] == current.get("endpoint", "auto")), 1),
    )
    mode = _choose(
        "Sidecar purpose",
        [("route", "route advisory"), ("verify", "verification review"), ("structured", "generic structured specialist")],
        next((i for i, item in enumerate(("route", "verify", "structured"), 1) if item == current.get("mode")), 3),
    )
    current.update({
        "model_id": model_id,
        "mode": mode,
        "endpoint": endpoint,
        "enabled": True,
        "timeout_seconds": _prompt_float("Timeout seconds", float(current.get("timeout_seconds", 45)), 1, 600),
        "max_packet_bytes": _prompt_int("Maximum packet bytes", int(current.get("max_packet_bytes", 64_000)), 1_024, 256_000),
        "max_output_tokens": _prompt_int("Maximum output tokens", int(current.get("max_output_tokens", 2_048)), 64, 131_072),
    })
    current["system_prompt"] = _prompt("System prompt (optional)", str(current.get("system_prompt", "")))
    sidecars[sidecar_id] = current
    _write_yaml(config_dir / "sidecars.yaml", raw)
    print(f"Saved sidecar '{sidecar_id}' to {config_dir / 'sidecars.yaml'}.")
    print("Reference it from a workflow phase with: execution_kind: sidecar_call and sidecar: " + sidecar_id)


def configure_keys(config_dir: Path, registry: ModelRegistry) -> None:
    """Interactively save/remove provider keys in the OS credential store."""
    path = config_dir / "providers.env"
    legacy_values = parse_env_file(path, allowed_keys=ALLOWED_PROVIDER_KEYS) if path.exists() else {}
    key_options: set[str] = set()
    for provider in registry.providers.values():
        if provider.api_key_env:
            key_options.add(provider.api_key_env)
    for model in registry.models.values():
        if model.api_key_env:
            key_options.add(model.api_key_env)
        for endpoint in model.endpoints.values():
            if endpoint.api_key_env:
                key_options.add(endpoint.api_key_env)
    key_options.update(
        item for item in ALLOWED_PROVIDER_KEYS
        if item.endswith("_API_KEY")
    )
    secure_slots: dict[str, dict[str, str]] = {}
    secure_available = credential_store_available()
    if secure_available:
        for key in sorted(key_options & PROVIDER_SECRET_KEYS):
            try:
                secure_slots[key] = read_slots(key, config_dir)
            except CredentialStoreUnavailable:
                secure_available = False
                secure_slots = {}
                break
    choices = []
    for key in sorted(key_options):
        if secure_slots.get(key):
            state = "keyring slots: " + ", ".join(sorted(secure_slots[key]))
        elif legacy_values.get(key):
            state = "legacy file value"
        else:
            state = "missing"
        choices.append((key, state))
    selected = _choose("Provider credential to save or remove", choices)
    existing_slots = sorted(secure_slots.get(selected, {}))
    slot = _prompt("Key slot name", existing_slots[0] if existing_slots else "primary")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", slot):
        raise ValueError("credential slot names may contain letters, numbers, '.', '_' and '-'")
    if secure_available:
        print(f"Enter the value for {selected}/{slot}. Input is hidden and will be stored in the OS keyring.")
    else:
        print(
            f"No usable OS keyring was found. Input is hidden and will be stored only in {path} "
            "with mode 0600."
        )
    value = getpass.getpass(f"{selected} (blank removes it): ")
    if secure_available:
        if value:
            set_slot(selected, slot, value, config_dir)
            legacy_values.pop(selected, None)
            _write_provider_env(path, legacy_values)
            print(f"Saved {selected}/{slot} in the OS keyring ({backend_label()}); active requests rotate slots round-robin.")
        else:
            delete_slot(selected, slot, config_dir)
            legacy_values.pop(selected, None)
            _write_provider_env(path, legacy_values)
            print(f"Removed {selected}/{slot} from the OS keyring and legacy file.")
    else:
        if value:
            legacy_values[selected] = value
            print(f"Saved {selected} in the protected legacy credential file.")
        else:
            legacy_values.pop(selected, None)
            print(f"Removed {selected} from the protected legacy credential file.")
        _write_provider_env(path, legacy_values)
        print(f"Credential file mode: {oct(path.stat().st_mode & 0o777)}")


def import_environment_credentials(config_dir: Path, registry: ModelRegistry) -> None:
    """Import explicitly loaded API keys without displaying their values."""
    key_options = set(PROVIDER_SECRET_KEYS)
    for provider in registry.providers.values():
        if provider.api_key_env:
            key_options.add(provider.api_key_env)
    available_keys = sorted(key for key in key_options if os.environ.get(key))
    if not available_keys:
        print("No provider API keys are present in the current environment.")
        return
    print("Loaded environment credentials: " + ", ".join(available_keys))
    if input("Import these key names into the OS keyring? [y/N]: ").strip().lower() != "y":
        print("Nothing imported.")
        return
    if not credential_store_available():
        raise CredentialStoreUnavailable(
            "no usable OS keyring is available; use 'keys' to save one interactively"
        )
    for key in available_keys:
        set_slot(key, "imported", os.environ[key], config_dir)
    print(f"Imported {len(available_keys)} credentials into the OS keyring ({backend_label()}); imported slots participate in rotation.")


def refresh_catalogs(registry: ModelRegistry) -> None:
    """Fetch configured OpenAI-compatible catalogs for keys that are present."""
    discovered_any = False
    for provider_id, provider in sorted(registry.providers.items()):
        if provider.discovery.get("type") != "openai-models":
            continue
        if not provider.api_key_env or not os.environ.get(provider.api_key_env):
            print(f"{provider_id}: skipped (missing {provider.api_key_env or 'credential'})")
            continue
        endpoint_id = "openai" if "openai" in provider.endpoints else next(
            iter(provider.endpoints), None
        )
        if not endpoint_id:
            print(f"{provider_id}: skipped (no catalog endpoint)")
            continue
        try:
            digest, count = registry.discover_provider_catalog(
                provider_id, endpoint_id=endpoint_id, timeout=30
            )
        except Exception as exc:
            print(f"{provider_id}: discovery failed ({str(exc)[:160]})")
            continue
        discovered_any = True
        print(f"{provider_id}: saved {count} models (catalog {digest[:12]})")
    if not discovered_any:
        print("No credential-backed dynamic provider catalogs were refreshed.")


def _delete_saved(config_dir: Path, *, kind: str) -> None:
    filename = "sidecars.yaml" if kind == "sidecar" else "profiles.yaml"
    key = "sidecars" if kind == "sidecar" else "profiles"
    raw = _read_yaml(config_dir / filename)
    entries = raw.get(key) if isinstance(raw.get(key), dict) else {}
    if not entries:
        print(f"No saved {kind} configurations found.")
        return
    selected = _choose(f"Delete saved {kind}", [(str(item), "saved configuration") for item in sorted(entries)])
    if input(f"Delete '{selected}' permanently? [y/N]: ").strip().lower() != "y":
        print("Nothing deleted.")
        return
    del entries[selected]
    _write_yaml(config_dir / filename, raw)
    print(f"Deleted {kind} '{selected}'.")


def show_saved(config_dir: Path, provider_keys: tuple[str, ...]) -> None:
    print(f"\nConfig directory: {config_dir}")
    print("Loaded provider credentials: " + (", ".join(sorted(provider_keys)) or "none"))
    if credential_store_available():
        print("Credential backend: " + backend_label())
        for key in sorted(PROVIDER_SECRET_KEYS):
            names = slot_names(key, config_dir)
            if names:
                print(f"  {key} slots: {', '.join(names)}")
    print("\nMain models -- controller + recon/implementer/adversary/repairer")
    profiles = _read_yaml(config_dir / "profiles.yaml").get("profiles") or {}
    for profile_id, value in sorted(profiles.items()):
        print(f"  {profile_id}:")
        print(f"    controller: {value.get('controller_model', 'Claude/default')}")
        for role in _ROLES:
            role_value = value.get(role)
            if isinstance(role_value, dict):
                model = role_value.get("model", "(unset)")
                fallbacks = role_value.get("fallback_models") or []
                suffix = f" (fallbacks: {', '.join(fallbacks)})" if fallbacks else ""
            else:
                model = role_value or "(unset)"
                suffix = ""
            print(f"    {role}: {model}{suffix}")
    if not profiles:
        print("  (none)")

    print("\nSidecar & fastpath models -- coprocessor + workflow-triggered specialists")
    fastpath = _read_yaml(config_dir / "fastpath.yaml").get("fastpath") or {}
    if fastpath:
        print(f"  fastpath coprocessor: model={fastpath.get('model_id', '(unset)')} enabled={fastpath.get('enabled', True)} modes={fastpath.get('modes', [])}")
    else:
        print("  fastpath coprocessor: (not configured)")
    sidecars = _read_yaml(config_dir / "sidecars.yaml").get("sidecars") or {}
    for sidecar_id, value in sorted(sidecars.items()):
        print(f"  sidecar '{sidecar_id}': model={value.get('model_id')} mode={value.get('mode', 'structured')} enabled={value.get('enabled', True)} (dormant unless a workflow phase references it)")
    if not sidecars:
        print("  named sidecars: (none)")


def menu(config_dir: Path) -> int:
    registry, provider_keys = _load_registry(config_dir)
    while True:
        print("\nClaudeBrigade saved configuration")
        print("\nMain models -- controller + recon/implementer/adversary/repairer")
        print("  1. Create/edit inference profile (models, endpoints, fallbacks)")
        print("  2. Delete inference profile")
        print("\nSidecar & fastpath models -- coprocessor + workflow-triggered specialists")
        print("  3. Configure fastpath coprocessor (always-on route/verify model)")
        print("  4. Create/edit sidecar (dormant until a workflow phase references it)")
        print("  5. Delete sidecar")
        print("\nProviders & credentials")
        print("  6. Add/edit provider API key")
        print("  7. Import already-loaded environment keys")
        print("  8. Refresh provider model catalogs")
        print("\n  9. Show saved configuration")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()
        try:
            if choice == "1":
                configure_inference(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "2":
                _delete_saved(config_dir, kind="inference profile")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "3":
                configure_fastpath(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "4":
                configure_sidecar(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "5":
                _delete_saved(config_dir, kind="sidecar")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "6":
                configure_keys(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "7":
                import_environment_credentials(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "8":
                refresh_catalogs(registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "9":
                show_saved(config_dir, provider_keys)
            elif choice in {"q", "quit", "exit"}:
                return 0
            else:
                print("Choose 1-9 or q.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"Configuration was not saved: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactively manage saved ClaudeBrigade inference and sidecar configuration")
    parser.add_argument("command", nargs="?", choices=("menu", "keys", "import-env", "refresh", "sidecar", "inference", "fastpath", "show"), default="menu")
    parser.add_argument("--config-dir", help="BRIGADE_CONFIG_DIR override")
    args = parser.parse_args(argv)
    config_dir = _config_dir(args.config_dir)
    if args.command == "menu":
        return menu(config_dir)
    registry, provider_keys = _load_registry(config_dir)
    try:
        if args.command == "keys":
            configure_keys(config_dir, registry)
        elif args.command == "import-env":
            import_environment_credentials(config_dir, registry)
        elif args.command == "refresh":
            refresh_catalogs(registry)
        elif args.command == "sidecar":
            configure_sidecar(config_dir, registry)
        elif args.command == "inference":
            configure_inference(config_dir, registry)
        elif args.command == "fastpath":
            configure_fastpath(config_dir, registry)
        else:
            show_saved(config_dir, provider_keys)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
