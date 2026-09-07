"""Schema-aware migration for the operator-owned provider configuration."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROVIDER_SCHEMA_VERSION = 2
_ROUTER_OWNED_PROVIDER_FIELDS = (
    "display_name",
    "api_key_env",
    "max_concurrency_env",
    "discovery",
)


def _deep_merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> bool:
    changed = False
    for key, default in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(default)
            changed = True
        elif isinstance(target[key], dict) and isinstance(default, dict):
            changed = _deep_merge_missing(target[key], default) or changed
    return changed


def merge_missing_provider_defaults(
    user_config: dict[str, Any],
    bundled_config: dict[str, Any],
    *,
    fields: tuple[str, ...] = _ROUTER_OWNED_PROVIDER_FIELDS,
) -> bool:
    """Merge missing router-owned fields into existing provider entries.

    Provider entries that the operator removed or added are left untouched.
    Existing values, including disabled/custom provider settings, are never
    replaced. Nested discovery fields are filled individually so a partial
    operator-owned discovery block can be upgraded safely.
    """
    user_providers = user_config.get("providers")
    bundled_providers = bundled_config.get("providers")
    if not isinstance(user_providers, dict) or not isinstance(bundled_providers, dict):
        return False

    changed = False
    for provider_id, user_provider in user_providers.items():
        bundled_provider = bundled_providers.get(provider_id)
        if not isinstance(user_provider, dict) or not isinstance(bundled_provider, dict):
            continue
        defaults = {
            field: bundled_provider[field]
            for field in fields
            if field in bundled_provider
        }
        changed = _deep_merge_missing(user_provider, defaults) or changed
    old_version = user_config.get("provider_schema_version", 0)
    if not isinstance(old_version, int) or old_version < PROVIDER_SCHEMA_VERSION:
        user_config["provider_schema_version"] = PROVIDER_SCHEMA_VERSION
        changed = True
    return changed


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked configuration: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def migrate_provider_config(user_path: str | Path, bundled_path: str | Path) -> bool:
    """Apply the provider schema migration to one user-owned YAML file."""
    user_path = Path(user_path)
    bundled_path = Path(bundled_path)
    if not user_path.exists() or not bundled_path.exists():
        return False
    user = yaml.safe_load(user_path.read_text(encoding="utf-8")) or {}
    bundled = yaml.safe_load(bundled_path.read_text(encoding="utf-8")) or {}
    if not isinstance(user, dict) or not isinstance(bundled, dict):
        return False
    if not merge_missing_provider_defaults(user, bundled):
        return False
    _write_yaml(user_path, user)
    return True
