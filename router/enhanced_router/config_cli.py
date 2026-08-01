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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    registry.load_sidecar_profiles()
    registry.load_launch_presets()
    # Tolerant, not fail-closed: a broken reference anywhere in the saved
    # config (e.g. a profile pointing at a model that was since removed)
    # must not prevent the wizard from even starting -- the operator needs
    # a working menu to *fix* it. Runtime startup (get_registry()) is a
    # separate, unrelated call path and still calls _validate_cross_refs()
    # directly, so the running router stays fail-closed on invalid config.
    for diagnostic in registry.collect_config_diagnostics():
        print(f"Warning: saved '{diagnostic.section}' configuration has an issue: {diagnostic.message}")
    for warning in registry.profile_model_diversity_warnings():
        print(f"Warning: {warning}")
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


_PAGE_SIZE = 15


def _choose(label: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """Numbered picker over *options*, paginated for lists longer than a
    screenful. Numbers are absolute across pages (option 17 is always
    option 17, on whichever page it's currently shown), so a remembered or
    typed-ahead number never shifts meaning when the page changes."""
    if not options:
        raise RuntimeError(f"No choices are available for {label}.")
    total_pages = (len(options) - 1) // _PAGE_SIZE + 1
    page = ((default - 1) // _PAGE_SIZE) if 1 <= default <= len(options) else 0
    while True:
        start = page * _PAGE_SIZE
        end = min(start + _PAGE_SIZE, len(options))
        suffix = f" (page {page + 1}/{total_pages})" if total_pages > 1 else ""
        print(f"\n{label}{suffix}")
        for number in range(start, end):
            value, description = options[number]
            print(f"  {number + 1}. {value} — {description}")
        if total_pages > 1:
            print("  n) next page   p) previous page")
        raw = _prompt("Choose", str(default))
        lowered = raw.strip().lower()
        if total_pages > 1 and lowered == "n":
            page = min(page + 1, total_pages - 1)
            continue
        if total_pages > 1 and lowered == "p":
            page = max(page - 1, 0)
            continue
        try:
            index = int(raw)
        except ValueError:
            print("Choose one of the listed numbers, or n/p to change page.")
            continue
        if 1 <= index <= len(options):
            return options[index - 1][0]
        print("Choose one of the listed numbers.")


def _rank_query_matches(query: str, options: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rank *options* against *query*, best match first.

    A plain substring filter treats "starts with the query" the same as
    "the query happens to appear buried in the description" -- with a
    provider-discovered catalog of hundreds of models, that buries the
    model the operator is actually typing towards under noise. Rank
    instead: exact ID match, then ID-prefix, then ID-substring, then
    label-substring; ties keep the original (already-sorted-by-ID) order.
    """
    query = query.lower()
    if not query:
        return list(options)

    def _rank(item: tuple[str, str]) -> int:
        model_id, label = item
        model_id_lower = model_id.lower()
        if model_id_lower == query:
            return 0
        if model_id_lower.startswith(query):
            return 1
        if query in model_id_lower:
            return 2
        if query in label.lower():
            return 3
        return 4

    ranked = [(index, item) for index, item in enumerate(options) if _rank(item) < 4]
    ranked.sort(key=lambda pair: (_rank(pair[1]), pair[0]))
    return [item for _, item in ranked]


def _distinct_providers(options: list[tuple[str, str]]) -> list[str]:
    """Extract each distinct 'provider=X' token from _model_choices labels,
    in first-seen order, for the provider-first filtering step."""
    seen: list[str] = []
    for _, label in options:
        for segment in label.split("; "):
            if segment.startswith("provider="):
                provider = segment[len("provider="):]
                if provider not in seen:
                    seen.append(provider)
    return seen


def _choose_model(
    label: str, options: list[tuple[str, str]], default_id: str = ""
) -> str:
    """Search a potentially large discovered catalog before choosing.

    Provider-first filtering narrows hundreds of discovered models down to
    one provider's handful before the operator has to type anything, and
    ranked search (see _rank_query_matches) puts the closest ID match
    first instead of preserving arbitrary catalog order.
    """
    providers = _distinct_providers(options)
    scoped = options
    if len(providers) > 1:
        provider_choice = input(
            f"Filter {label} by provider ({', '.join(providers)}; blank = all): "
        ).strip()
        if provider_choice:
            narrowed = [item for item in options if f"provider={provider_choice}" in item[1]]
            if narrowed:
                scoped = narrowed
            else:
                print(f"No models from provider '{provider_choice}'; showing all providers instead.")
    while True:
        query = input(f"Search {label} (blank lists all): ").strip().lower()
        filtered = scoped if not query else _rank_query_matches(query, scoped)
        if not filtered:
            print(f"No matches for '{query}'. Press Enter with nothing typed to list all {len(scoped)} options.")
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
        fallback_ids = []
        for item in raw.get("fallback_models", []):
            if isinstance(item, str):
                fallback_ids.append(item)
            elif isinstance(item, dict) and isinstance(item.get("model"), str):
                fallback_ids.append(item["model"])
        return str(raw.get("model", "")), str(raw.get("endpoint", "auto")), fallback_ids
    return str(raw), "auto", []


def _profile_fallback_routes(profile: dict[str, Any], role: str) -> list[dict]:
    """Like _profile_model's third element, but keeps each fallback's own
    endpoint (not just its model ID) for handing to _edit_fallbacks."""
    raw = profile.get(role, "")
    if not isinstance(raw, dict):
        return []
    routes = []
    for item in raw.get("fallback_models", []):
        if isinstance(item, str):
            routes.append({"model": item, "endpoint": "auto"})
        elif isinstance(item, dict) and isinstance(item.get("model"), str):
            routes.append({"model": item["model"], "endpoint": str(item.get("endpoint", "auto"))})
    return routes


def _edit_fallbacks(
    config_dir: Path,
    registry: ModelRegistry,
    *,
    role: str | None,
    controller: bool = False,
    fallback_options: list[tuple[str, str]],
    previous: list[dict],
) -> list[dict]:
    """Interactive add/remove/reorder editor for a route's fallback ladder.

    Each fallback is a full {model, endpoint} pair (RouteCandidateSpec): a
    fallback can pin a different provider/endpoint than the primary route,
    not just fall back to the same model under 'auto'. Replaces a bare
    comma-separated free-text prompt, which had no way to express a
    per-fallback endpoint, reorder the ladder, or browse/search candidates.
    """
    fallbacks = [dict(item) for item in previous if isinstance(item, dict) and item.get("model")]
    if not fallback_options:
        return fallbacks
    target = _target_label(role, controller)
    while True:
        print(f"\nFallback ladder for {target}:")
        if fallbacks:
            for index, item in enumerate(fallbacks, 1):
                print(f"  {index}. {item['model']} (endpoint: {item.get('endpoint', 'auto')})")
        else:
            print("  (none)")
        action = input("a) add fallback   r) remove   m) move   d) done: ").strip().lower()
        if action == "a":
            already_used = {item["model"] for item in fallbacks}
            candidates = [item for item in fallback_options if item[0] not in already_used]
            if not candidates:
                print("No more distinct models are available to add.")
                continue
            model_id = _choose_model(f"Fallback #{len(fallbacks) + 1} for {target}", candidates)
            if not _confirm_and_grant(config_dir, registry, model_id, role=role, controller=controller):
                print(f"Dropping '{model_id}' (not confirmed).")
                continue
            model = registry.get_model(model_id)
            endpoints = [("auto", "registry endpoint selection")]
            endpoints.extend(
                (endpoint_id, f"configured {endpoint.backend} endpoint")
                for endpoint_id, endpoint in sorted(model.endpoints.items())
            )
            endpoint = _choose(f"Endpoint for fallback '{model_id}'", endpoints)
            fallbacks.append({"model": model_id, "endpoint": endpoint})
        elif action == "r":
            if not fallbacks:
                print("No fallbacks to remove.")
                continue
            index = _prompt_int("Remove which number", 1, 1, len(fallbacks))
            removed = fallbacks.pop(index - 1)
            print(f"Removed '{removed['model']}'.")
        elif action == "m":
            if len(fallbacks) < 2:
                print("Need at least two fallbacks to reorder.")
                continue
            source = _prompt_int("Move which number", 1, 1, len(fallbacks))
            destination = _prompt_int("Move to which position", source, 1, len(fallbacks))
            item = fallbacks.pop(source - 1)
            fallbacks.insert(destination - 1, item)
        elif action == "d":
            return fallbacks
        else:
            print("Choose a, r, m, or d.")


def _choose_or_create_id(kind: str, existing: list[str], default_new: str) -> str:
    """Numbered picker over existing saved *kind* entries, with a 'create
    new' option -- replaces a free-text name prompt that silently defaulted
    to the first existing entry and gave no browsable list beyond one
    printed line.
    """
    if not existing:
        name = _prompt(f"{kind} name", default_new)
    else:
        options = [(item, "existing") for item in existing] + [("(new)", "create a new one")]
        choice = _choose(f"Select a {kind} to edit, or create a new one", options)
        name = _prompt(f"New {kind} name", default_new) if choice == "(new)" else choice
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"{kind} IDs may contain letters, numbers, '.', '_' and '-'")
    return name


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
    if not existing:
        print("No profiles saved yet -- name this one (letters, numbers, '.', '_', '-').")
    profile_id = _choose_or_create_id("Inference profile", existing, "my-profile")
    current = dict(profiles.get(profile_id) or {})
    print("\n=== Main models: controller + recon/implementer/adversary/repairer ===")
    print("For each, type part of a name to search, or press Enter to list everything.")
    controller_default = str(current.get("controller_model") or (current.get("controller") or {}).get("model") or "")
    controller_choices = _model_choices(registry, controller=True)
    if controller_choices:
        while True:
            controller_model = _choose_model(
                "Controller model", controller_choices,
                controller_default,
            )
            if _confirm_and_grant(config_dir, registry, controller_model, controller=True):
                break
        controller_fallback_options = [item for item in controller_choices if item[0] != controller_model]
        controller_fallbacks = _edit_fallbacks(
            config_dir, registry, role=None, controller=True,
            fallback_options=controller_fallback_options,
            previous=_profile_fallback_routes(current, "controller"),
        )
        current.pop("controller_model", None)
        current.pop("controller", None)
        if controller_fallbacks:
            current["controller"] = {
                "model": controller_model, "endpoint": "auto",
                "fallback_models": controller_fallbacks,
            }
        else:
            current["controller_model"] = controller_model
    for role in _ROLES:
        choices = _model_choices(registry, role=role)
        if not choices:
            raise RuntimeError(f"No credential-backed model is available for role '{role}'.")
        previous, previous_endpoint, _previous_fallbacks = _profile_model(current, role)
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
        fallback_routes = _edit_fallbacks(
            config_dir, registry, role=role,
            fallback_options=fallback_options,
            previous=_profile_fallback_routes(current, role),
        )
        if endpoint == "auto" and not fallback_routes:
            current[role] = selected
        else:
            current[role] = {
                "model": selected,
                "endpoint": endpoint,
                "fallback_models": fallback_routes,
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
    sidecar_id = _choose_or_create_id("Sidecar", existing, "verification_reviewer")
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


def configure_sidecar_profile(config_dir: Path, registry: ModelRegistry) -> str:
    """Configure a named, reusable sidecar-profile bundle (sidecar_profiles.yaml).

    A sidecar profile limits which globally-defined sidecars (sidecars.yaml)
    are available for a launch, and carries its own fastpath coprocessor
    config -- fastpath is scoped per sidecar-profile, not one bare global
    singleton, so different launch presets can run different fastpath
    models (or none) side by side.
    """
    print("\n=== Sidecar & fastpath models: sidecar profile (bounds which sidecars a launch may use) ===")
    raw = _read_yaml(config_dir / "sidecar_profiles.yaml")
    profiles = raw.setdefault("sidecar_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        raw["sidecar_profiles"] = profiles
    existing = sorted(str(item) for item in profiles)
    profile_id = _choose_or_create_id("Sidecar profile", existing, "default")
    current = dict(profiles.get(profile_id) or {})

    available_sidecars = sorted(registry.sidecars)
    if not available_sidecars:
        raise RuntimeError("No sidecars are configured yet -- add one with the 'sidecar' menu option first.")
    print(f"Available sidecars: {', '.join(available_sidecars)}")
    previous_ids = [str(item) for item in current.get("sidecar_ids", []) if isinstance(item, str)]
    raw_ids = _prompt(
        "Sidecar IDs for this profile (comma-separated, blank for all)",
        ",".join(previous_ids),
    )
    if raw_ids.strip():
        sidecar_ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
        unknown = [item for item in sidecar_ids if item not in registry.sidecars]
        if unknown:
            raise ValueError(f"unknown sidecar ID(s): {', '.join(unknown)}")
    else:
        sidecar_ids = available_sidecars
    current["sidecar_ids"] = sidecar_ids

    if input("Configure a dedicated fastpath coprocessor for this profile too? [y/N]: ").strip().lower() == "y":
        fastpath = dict(current.get("fastpath") or {})
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
        fastpath["timeout_seconds"] = _prompt_float(
            "Fastpath timeout seconds", float(fastpath.get("timeout_seconds", 5)), 1, 30,
        )
        current["fastpath"] = fastpath
    elif "fastpath" in current and input(
        "Remove this profile's dedicated fastpath (fall back to the global fastpath.yaml)? [y/N]: "
    ).strip().lower() == "y":
        current.pop("fastpath", None)

    profiles[profile_id] = current
    _write_yaml(config_dir / "sidecar_profiles.yaml", raw)
    print(f"Saved sidecar profile '{profile_id}' to {config_dir / 'sidecar_profiles.yaml'}.")
    print(f"Use it with: claude-brigade --sidecar-profile {profile_id}")
    return profile_id


def configure_launch_preset(config_dir: Path, registry: ModelRegistry) -> str:
    """Configure a named launch preset pairing a saved inference profile with
    a saved sidecar profile, so both switch together with one choice."""
    print("\n=== Launch presets: pair a saved inference profile with a saved sidecar profile ===")
    inference_profiles = sorted(_read_yaml(config_dir / "profiles.yaml").get("profiles") or {})
    if not inference_profiles:
        raise RuntimeError("No inference profiles are saved yet -- run the inference wizard first.")
    sidecar_profiles = sorted(_read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {})

    raw = _read_yaml(config_dir / "launch_presets.yaml")
    presets = raw.setdefault("launch_presets", {})
    if not isinstance(presets, dict):
        presets = {}
        raw["launch_presets"] = presets
    existing = sorted(str(item) for item in presets)
    preset_id = _choose_or_create_id("Launch preset", existing, "default")
    current = dict(presets.get(preset_id) or {})

    inference_profile_id = _choose(
        "Inference profile",
        [(item, "saved inference profile") for item in inference_profiles],
        next((i for i, item in enumerate(inference_profiles, 1) if item == current.get("inference_profile_id")), 1),
    )
    sidecar_profile_options = [("none", "no sidecar profile (use global sidecars.yaml/fastpath.yaml)")] + [
        (item, "saved sidecar profile") for item in sidecar_profiles
    ]
    sidecar_profile_choice = _choose(
        "Sidecar profile",
        sidecar_profile_options,
        next((i for i, item in enumerate(("none", *sidecar_profiles), 1) if item == (current.get("sidecar_profile_id") or "none")), 1),
    )
    current["inference_profile_id"] = inference_profile_id
    current["sidecar_profile_id"] = None if sidecar_profile_choice == "none" else sidecar_profile_choice

    presets[preset_id] = current
    _write_yaml(config_dir / "launch_presets.yaml", raw)
    print(f"Saved launch preset '{preset_id}' to {config_dir / 'launch_presets.yaml'}.")
    print(f"Use it with: claude-brigade --launch-preset {preset_id}")
    return preset_id


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


_CATALOG_REFRESH_TTL_SECONDS = 900.0  # 15 minutes
_CATALOG_REFRESH_TIMEOUT_SECONDS = 12.0
_CATALOG_REFRESH_MAX_WORKERS = 4
_CATALOG_REFRESH_STATE_FILENAME = "catalog_refresh_state.json"


@dataclass(frozen=True)
class CatalogRefreshResult:
    """One provider's outcome from a ``refresh_catalogs`` pass."""

    provider_id: str
    status: Literal["success", "skipped", "cached", "error"]
    model_count: int = 0
    digest: str | None = None
    detail: str = ""


def _load_catalog_refresh_state(config_dir: Path) -> dict[str, dict[str, Any]]:
    path = config_dir / _CATALOG_REFRESH_STATE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_catalog_refresh_state(config_dir: Path, state: dict[str, dict[str, Any]]) -> None:
    path = config_dir / _CATALOG_REFRESH_STATE_FILENAME
    path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)


def refresh_catalogs(
    registry: ModelRegistry,
    *,
    ttl_seconds: float = _CATALOG_REFRESH_TTL_SECONDS,
    force: bool = False,
) -> list[CatalogRefreshResult]:
    """Fetch configured OpenAI-compatible catalogs for keys that are present.

    Network fetches for distinct providers run concurrently (bounded pool,
    a short per-provider timeout) instead of serially at up to 30s each --
    the previous behavior meant a config with several credential-backed
    providers could block the wizard for minutes even when every provider
    responds quickly on its own. A provider refreshed within *ttl_seconds*
    is skipped (``force=True`` bypasses this) so re-entering the wizard
    doesn't repeat unnecessary network round-trips.
    """
    config_dir = registry.config_dir
    refresh_state = _load_catalog_refresh_state(config_dir) if config_dir else {}
    now = time.time()

    candidates: list[tuple[str, str]] = []  # (provider_id, endpoint_id)
    results: list[CatalogRefreshResult] = []
    for provider_id, provider in sorted(registry.providers.items()):
        if provider.discovery.get("type") != "openai-models":
            continue
        if not provider.api_key_env or not os.environ.get(provider.api_key_env):
            results.append(CatalogRefreshResult(
                provider_id, "skipped",
                detail=f"missing {provider.api_key_env or 'credential'}",
            ))
            continue
        if not force:
            last_refreshed = refresh_state.get(provider_id, {}).get("refreshed_at")
            if isinstance(last_refreshed, (int, float)) and now - last_refreshed < ttl_seconds:
                results.append(CatalogRefreshResult(
                    provider_id, "cached",
                    detail=f"refreshed {int(now - last_refreshed)}s ago",
                ))
                continue
        endpoint_id = provider.discovery.get("endpoint_id") or (
            "openai" if "openai" in provider.endpoints else next(iter(provider.endpoints), None)
        )
        if not endpoint_id and not provider.discovery.get("url"):
            results.append(CatalogRefreshResult(provider_id, "skipped", detail="no catalog endpoint"))
            continue
        candidates.append((provider_id, endpoint_id or "openai"))

    fetched: dict[str, Any] = {}
    errors: dict[str, str] = {}
    if candidates:
        with ThreadPoolExecutor(max_workers=min(_CATALOG_REFRESH_MAX_WORKERS, len(candidates))) as pool:
            futures = {
                pool.submit(
                    registry.fetch_discovered_entries,
                    provider_id, endpoint_id=endpoint_id, timeout=_CATALOG_REFRESH_TIMEOUT_SECONDS,
                ): provider_id
                for provider_id, endpoint_id in candidates
            }
            for future in as_completed(futures):
                provider_id = futures[future]
                try:
                    fetched[provider_id] = future.result()
                except Exception as exc:
                    errors[provider_id] = str(exc)[:160]

    # Publishing mutates shared registry state (the discovered-catalog file
    # and in-memory models) -- serialize it on the main thread even though
    # the fetches above ran concurrently.
    for provider_id, _endpoint_id in candidates:
        if provider_id in errors:
            results.append(CatalogRefreshResult(provider_id, "error", detail=errors[provider_id]))
            continue
        entries = fetched.get(provider_id, [])
        try:
            digest, count = registry.apply_discovered_entries(entries, provider_id=provider_id)
        except Exception as exc:
            results.append(CatalogRefreshResult(provider_id, "error", detail=str(exc)[:160]))
            continue
        results.append(CatalogRefreshResult(provider_id, "success", model_count=count, digest=digest))
        refresh_state[provider_id] = {"refreshed_at": now, "digest": digest}

    if config_dir and any(r.status == "success" for r in results):
        _save_catalog_refresh_state(config_dir, refresh_state)

    for result in sorted(results, key=lambda r: r.provider_id):
        if result.status == "success":
            print(f"{result.provider_id}: saved {result.model_count} models (catalog {(result.digest or '')[:12]})")
        elif result.status == "cached":
            print(f"{result.provider_id}: skipped ({result.detail}, within TTL)")
        elif result.status == "skipped":
            print(f"{result.provider_id}: skipped ({result.detail})")
        else:
            print(f"{result.provider_id}: discovery failed ({result.detail})")
    if not any(r.status == "success" for r in results):
        print("No credential-backed dynamic provider catalogs were refreshed.")
    return results


_DELETE_TARGETS = {
    "sidecar": ("sidecars.yaml", "sidecars"),
    "sidecar profile": ("sidecar_profiles.yaml", "sidecar_profiles"),
    "launch preset": ("launch_presets.yaml", "launch_presets"),
}


def _referencing_configs(config_dir: Path, kind: str, item_id: str) -> list[str]:
    """Return human-readable descriptions of saved configs that still
    reference *item_id* -- deleting it out from under them would leave a
    dangling reference that only surfaces later, as a fail-closed daemon
    startup error for an operator who may not remember this deletion.
    """
    blockers: list[str] = []
    if kind == "sidecar":
        sidecar_profiles = _read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {}
        for profile_id, profile in sidecar_profiles.items():
            if isinstance(profile, dict) and item_id in (profile.get("sidecar_ids") or []):
                blockers.append(f"sidecar profile '{profile_id}'")
    elif kind == "sidecar profile":
        launch_presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
        for preset_id, preset in launch_presets.items():
            if isinstance(preset, dict) and preset.get("sidecar_profile_id") == item_id:
                blockers.append(f"launch preset '{preset_id}'")
    elif kind == "inference profile":
        launch_presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
        for preset_id, preset in launch_presets.items():
            if isinstance(preset, dict) and preset.get("inference_profile_id") == item_id:
                blockers.append(f"launch preset '{preset_id}'")
        workflows = _read_yaml(config_dir / "workflows.yaml").get("workflows") or {}
        for workflow_id, workflow in workflows.items():
            if isinstance(workflow, dict) and workflow.get("default_profile") == item_id:
                blockers.append(f"workflow '{workflow_id}'")
    return blockers


def _delete_saved(config_dir: Path, *, kind: str) -> None:
    filename, key = _DELETE_TARGETS.get(kind, ("profiles.yaml", "profiles"))
    raw = _read_yaml(config_dir / filename)
    entries = raw.get(key) if isinstance(raw.get(key), dict) else {}
    if not entries:
        print(f"No saved {kind} configurations found.")
        return
    selected = _choose(f"Delete saved {kind}", [(str(item), "saved configuration") for item in sorted(entries)])
    blockers = _referencing_configs(config_dir, kind, selected)
    if blockers:
        print(f"Cannot delete '{selected}': still referenced by {', '.join(blockers)}.")
        return
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
        controller_raw = value.get("controller")
        if isinstance(controller_raw, dict):
            controller_label = controller_raw.get("model", "(unset)")
        else:
            controller_label = value.get("controller_model") or "Claude/default"
        print(f"    controller: {controller_label}")
        for role in _ROLES:
            role_value = value.get(role)
            if isinstance(role_value, dict):
                model = role_value.get("model", "(unset)")
                fallback_ids = [
                    item if isinstance(item, str) else item.get("model", "")
                    for item in (role_value.get("fallback_models") or [])
                    if isinstance(item, (str, dict))
                ]
                fallback_ids = [item for item in fallback_ids if item]
                suffix = f" (fallbacks: {', '.join(fallback_ids)})" if fallback_ids else ""
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

    sidecar_profiles = _read_yaml(config_dir / "sidecar_profiles.yaml").get("sidecar_profiles") or {}
    for profile_id, value in sorted(sidecar_profiles.items()):
        sidecar_ids = value.get("sidecar_ids") or []
        own_fastpath = value.get("fastpath")
        fastpath_label = f"model={own_fastpath.get('model_id')}" if isinstance(own_fastpath, dict) else "(uses global fastpath.yaml)"
        print(f"  sidecar profile '{profile_id}': sidecars=[{', '.join(sidecar_ids)}] fastpath={fastpath_label}")
    if not sidecar_profiles:
        print("  sidecar profiles: (none)")

    presets = _read_yaml(config_dir / "launch_presets.yaml").get("launch_presets") or {}
    for preset_id, value in sorted(presets.items()):
        print(
            f"  launch preset '{preset_id}': inference_profile={value.get('inference_profile_id')} "
            f"sidecar_profile={value.get('sidecar_profile_id') or '(none)'}"
        )
    if not presets:
        print("  launch presets: (none)")


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
        print("  6. Create/edit sidecar profile (bounds which sidecars a launch may use)")
        print("  7. Delete sidecar profile")
        print("\nLaunch presets -- pair a saved inference profile with a saved sidecar profile")
        print("  8. Create/edit launch preset")
        print("  9. Delete launch preset")
        print("\nProviders & credentials")
        print("  10. Add/edit provider API key")
        print("  11. Import already-loaded environment keys")
        print("  12. Refresh provider model catalogs")
        print("\n  13. Show saved configuration")
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
                configure_sidecar_profile(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "7":
                _delete_saved(config_dir, kind="sidecar profile")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "8":
                configure_launch_preset(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "9":
                _delete_saved(config_dir, kind="launch preset")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "10":
                configure_keys(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "11":
                import_environment_credentials(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "12":
                refresh_catalogs(registry, force=True)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "13":
                show_saved(config_dir, provider_keys)
            elif choice in {"q", "quit", "exit"}:
                return 0
            else:
                print("Choose 1-13 or q.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"Configuration was not saved: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactively manage saved ClaudeBrigade inference and sidecar configuration")
    parser.add_argument(
        "command", nargs="?",
        choices=(
            "menu", "keys", "import-env", "refresh", "sidecar", "inference",
            "fastpath", "sidecar-profile", "launch-preset", "show",
        ),
        default="menu",
    )
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
            refresh_catalogs(registry, force=True)
        elif args.command == "sidecar":
            configure_sidecar(config_dir, registry)
        elif args.command == "inference":
            configure_inference(config_dir, registry)
        elif args.command == "fastpath":
            configure_fastpath(config_dir, registry)
        elif args.command == "sidecar-profile":
            configure_sidecar_profile(config_dir, registry)
        elif args.command == "launch-preset":
            configure_launch_preset(config_dir, registry)
        else:
            show_saved(config_dir, provider_keys)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
