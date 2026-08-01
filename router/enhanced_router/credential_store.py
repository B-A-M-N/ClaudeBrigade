"""OS-backed credential storage for ClaudeBrigade provider keys.

The router should not require provider credentials to live in a project file
or in the Claude Code process environment.  ``keyring`` is optional at import
time so development and legacy installations can still start; when it is
available, the CLI uses the user's desktop/server credential backend and the
router reads keys directly in its own process.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from threading import Lock

import yaml

SERVICE_NAME = "claude-brigade"
SLOTS_FILE = "credential_slots.yaml"
_ROTATION_LOCK = Lock()
_ROTATION_INDEX: dict[str, int] = {}


class CredentialStoreUnavailable(RuntimeError):
    """Raised when no usable OS credential backend is available."""


def _backend():
    try:
        keyring = importlib.import_module("keyring")
        keyring_errors = importlib.import_module("keyring.errors")
        KeyringError = keyring_errors.KeyringError
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise CredentialStoreUnavailable(
            "Python keyring is not installed; use the locked providers.env fallback"
        ) from exc

    backend = keyring.get_keyring()
    backend_name = type(backend).__name__.lower()
    module_name = type(backend).__module__.lower()
    if backend_name == "failkeyring" or ".fail" in module_name:
        raise CredentialStoreUnavailable(
            "no usable OS credential backend is configured"
        )
    return keyring, KeyringError


def available() -> bool:
    """Return whether an actual OS-backed keyring can be used."""
    try:
        _backend()
    except Exception:
        return False
    return True


def backend_label() -> str:
    """Return a non-secret diagnostic label for the active backend."""
    keyring, _ = _backend()
    backend = keyring.get_keyring()
    return f"{type(backend).__module__}.{type(backend).__name__}"


def read(keys: Iterable[str]) -> dict[str, str]:
    """Read non-empty credentials from the OS keyring without logging values."""
    keyring, keyring_error = _backend()
    result: dict[str, str] = {}
    for key in sorted(set(keys)):
        try:
            value = keyring.get_password(SERVICE_NAME, key)
        except keyring_error as exc:
            raise CredentialStoreUnavailable(
                f"credential backend failed while reading {key}"
            ) from exc
        if value:
            result[key] = value
    return result


def _config_dir(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    configured = os.environ.get("BRIGADE_CONFIG_DIR")
    if configured:
        return Path(configured)
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "claude-brigade"


def _slot_manifest(config_dir: str | Path | None = None) -> dict[str, dict]:
    path = _config_dir(config_dir) / SLOTS_FILE
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    credentials = data.get("credentials") if isinstance(data, dict) else {}
    return credentials if isinstance(credentials, dict) else {}


def _account_name(key: str, slot: str) -> str:
    return key if slot == "default" else f"{key}::{slot}"


def materialized_env_name(key: str, slot: str) -> str:
    """Return a stable router/LiteLLM-only env name for one key slot."""
    digest = hashlib.sha256(f"{key}::{slot}".encode()).hexdigest()[:20].upper()
    return f"BRIGADE_KEYRING_{digest}"


def _manifest_slots(key: str, config_dir: str | Path | None = None) -> list[str]:
    raw = _slot_manifest(config_dir).get(key) or {}
    slots = raw.get("slots") if isinstance(raw, dict) else []
    if not isinstance(slots, list):
        return []
    result = [str(item).strip() for item in slots if str(item).strip()]
    return list(dict.fromkeys(result))


def read_slots(key: str, config_dir: str | Path | None = None) -> dict[str, str]:
    """Read all named keyring slots for one provider env name."""
    keyring, keyring_error = _backend()
    slots = _manifest_slots(key, config_dir) or ["default"]
    values: dict[str, str] = {}
    for slot in slots:
        try:
            value = keyring.get_password(SERVICE_NAME, _account_name(key, slot))
        except keyring_error as exc:
            raise CredentialStoreUnavailable(
                f"credential backend failed while reading {key}/{slot}"
            ) from exc
        if value:
            values[slot] = value
    return values


def resolve(key: str, config_dir: str | Path | None = None) -> str | None:
    """Return the next active key for a provider using round-robin rotation."""
    try:
        values = read_slots(key, config_dir)
    except CredentialStoreUnavailable:
        return None
    if not values:
        return None
    slots = list(values)
    with _ROTATION_LOCK:
        index = _ROTATION_INDEX.get(key, 0) % len(slots)
        _ROTATION_INDEX[key] = index + 1
    return values[slots[index]]


def resolve_loaded(key: str, config_dir: str | Path | None = None) -> str | None:
    """Rotate credentials already materialized in the router process.

    Request paths use this non-blocking variant instead of opening the OS
    keyring on every request. The keyring is read once during router startup;
    only generated router-local environment names are consulted afterward.
    """
    slots = _manifest_slots(key, config_dir) or ["default"]
    values = {
        slot: os.environ.get(materialized_env_name(key, slot), "")
        for slot in slots
    }
    values = {slot: value for slot, value in values.items() if value}
    if not values:
        return os.environ.get(key) or None
    active_slots = list(values)
    with _ROTATION_LOCK:
        index = _ROTATION_INDEX.get(f"loaded:{key}", 0) % len(active_slots)
        _ROTATION_INDEX[f"loaded:{key}"] = index + 1
    return values[active_slots[index]]


def all_materialized(config_dir: str | Path | None, keys: Iterable[str]) -> dict[str, str]:
    """Return slot values under generated names for the router/LiteLLM child."""
    result: dict[str, str] = {}
    for key in sorted(set(keys)):
        try:
            slots = read_slots(key, config_dir)
        except CredentialStoreUnavailable:
            continue
        for slot, value in slots.items():
            result[materialized_env_name(key, slot)] = value
    return result


def slot_names(key: str, config_dir: str | Path | None = None) -> tuple[str, ...]:
    return tuple(_manifest_slots(key, config_dir))


def set_slot(
    key: str,
    slot: str,
    value: str,
    config_dir: str | Path | None = None,
) -> None:
    """Store a named slot and update only its non-secret manifest metadata."""
    slot = slot.strip()
    if not slot or any(character.isspace() for character in slot) or "::" in slot:
        raise ValueError("credential slot names must be non-empty and contain no whitespace or '::'")
    set_value(_account_name(key, slot), value)
    manifest = _slot_manifest(config_dir)
    entry = dict(manifest.get(key) or {})
    slots = _manifest_slots(key, config_dir)
    if slot not in slots:
        slots.append(slot)
    entry["slots"] = slots
    entry["rotation"] = "round_robin"
    manifest[key] = entry
    _write_slot_manifest(_config_dir(config_dir), manifest)


def delete_slot(key: str, slot: str, config_dir: str | Path | None = None) -> None:
    """Delete one named slot and remove it from the non-secret manifest."""
    delete(_account_name(key, slot))
    manifest = _slot_manifest(config_dir)
    entry = dict(manifest.get(key) or {})
    slots = [item for item in _manifest_slots(key, config_dir) if item != slot]
    if slots:
        entry["slots"] = slots
        entry["rotation"] = "round_robin"
        manifest[key] = entry
    else:
        manifest.pop(key, None)
    _write_slot_manifest(_config_dir(config_dir), manifest)


def _write_slot_manifest(config_dir: Path, manifest: dict[str, dict]) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / SLOTS_FILE
    if path.is_symlink():
        raise CredentialStoreUnavailable(f"refusing to replace symlinked {SLOTS_FILE}")
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(config_dir), text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump({"credentials": manifest}, sort_keys=False))
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def set_value(key: str, value: str) -> None:
    """Store one credential in the OS keyring."""
    keyring, keyring_error = _backend()
    try:
        keyring.set_password(SERVICE_NAME, key, value)
    except keyring_error as exc:
        raise CredentialStoreUnavailable(
            f"credential backend failed while storing {key}"
        ) from exc


def delete(key: str) -> None:
    """Delete one credential, treating an absent entry as already deleted."""
    keyring, keyring_error = _backend()
    try:
        keyring.delete_password(SERVICE_NAME, key)
    except Exception as exc:
        if type(exc).__name__ == "PasswordDeleteError":
            return
        if isinstance(exc, keyring_error):
            raise CredentialStoreUnavailable(
                f"credential backend failed while deleting {key}"
            ) from exc
        raise
