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
            print("No matching models. Try another search.")
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
    choices: list[tuple[str, str]] = []
    source = registry.controller_models() if controller else registry.models.items()
    for model_id, model in sorted(source, key=lambda item: item[0]):
        if not model.enabled or model.backend == "anthropic-passthrough" and not controller:
            continue
        if role is not None and role not in model.allowed_roles:
            continue
        provider, backend, key_env, configured = _model_provider(registry, model_id)
        if not configured:
            continue
        label = f"{model.display_name}; {backend}; provider={provider}; key={key_env}"
        choices.append((model_id, label))
    return choices


def _profile_model(profile: dict[str, Any], role: str) -> tuple[str, str, list[str]]:
    raw = profile.get(role, "")
    if isinstance(raw, dict):
        return (
            str(raw.get("model", "")),
            str(raw.get("endpoint", "auto")),
            [str(item) for item in raw.get("fallback_models", []) if isinstance(item, str)],
        )
    return str(raw), "auto", []


def configure_inference(config_dir: Path, registry: ModelRegistry) -> None:
    raw = _read_yaml(config_dir / "profiles.yaml")
    profiles = raw.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        raw["profiles"] = profiles
    existing = sorted(str(item) for item in profiles)
    profile_id = _prompt("Saved inference profile ID", existing[0] if existing else "my-profile")
    if not _NAME_RE.fullmatch(profile_id):
        raise ValueError("profile IDs may contain letters, numbers, '.', '_' and '-'")
    current = dict(profiles.get(profile_id) or {})
    print("\nSelect the models ClaudeBrigade's controller and native roles should use.")
    controller_default = str(current.get("controller_model") or "")
    controller_choices = _model_choices(registry, controller=True)
    if controller_choices:
        controller_model = _choose_model(
            "Controller model", controller_choices,
            controller_default,
        )
        current["controller_model"] = controller_model
    for role in _ROLES:
        choices = _model_choices(registry, role=role)
        if not choices:
            raise RuntimeError(f"No credential-backed model is available for role '{role}'.")
        previous, previous_endpoint, previous_fallbacks = _profile_model(current, role)
        selected = _choose_model(
            f"{role} model", choices,
            previous,
        )
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


def configure_sidecar(config_dir: Path, registry: ModelRegistry) -> None:
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
    for kind, filename, key in (("sidecar", "sidecars.yaml", "sidecars"), ("inference profile", "profiles.yaml", "profiles")):
        entries = _read_yaml(config_dir / filename).get(key) or {}
        print(f"\nSaved {kind}s:")
        for entry_id, value in sorted(entries.items()):
            if kind == "sidecar":
                print(f"  {entry_id}: model={value.get('model_id')} mode={value.get('mode', 'structured')} enabled={value.get('enabled', True)}")
            else:
                print(f"  {entry_id}: controller={value.get('controller_model', 'Claude/default')}")
        if not entries:
            print("  (none)")


def menu(config_dir: Path) -> int:
    registry, provider_keys = _load_registry(config_dir)
    while True:
        print("\nClaudeBrigade saved configuration")
        print("  1. Add/edit provider API key")
        print("  2. Import already-loaded environment keys")
        print("  3. Create/edit sidecar")
        print("  4. Create/edit inference profile")
        print("  5. Delete sidecar")
        print("  6. Delete inference profile")
        print("  7. Refresh provider model catalogs")
        print("  8. Show saved configurations")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()
        try:
            if choice == "1":
                configure_keys(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "2":
                import_environment_credentials(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "3":
                configure_sidecar(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "4":
                configure_inference(config_dir, registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "5":
                _delete_saved(config_dir, kind="sidecar")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "6":
                _delete_saved(config_dir, kind="inference profile")
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "7":
                refresh_catalogs(registry)
                registry, provider_keys = _load_registry(config_dir)
            elif choice == "8":
                show_saved(config_dir, provider_keys)
            elif choice in {"q", "quit", "exit"}:
                return 0
            else:
                print("Choose 1-8 or q.")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"Configuration was not saved: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactively manage saved ClaudeBrigade inference and sidecar configuration")
    parser.add_argument("command", nargs="?", choices=("menu", "keys", "import-env", "refresh", "sidecar", "inference", "show"), default="menu")
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
        else:
            show_saved(config_dir, provider_keys)
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
