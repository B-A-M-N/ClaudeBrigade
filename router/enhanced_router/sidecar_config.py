"""Non-destructive migration for native sidecar and coprocessor config."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml


def migrate_sidecar_config(user_path: str | Path, bundled_path: str | Path) -> bool:
    """Add missing bundled worker definitions without replacing operator policy.

    Existing ``sidecars`` entries are legacy bounded coprocessors and remain
    untouched. Native sidecars and new coprocessors are merged by identifier;
    an operator's existing definition always wins.
    """
    user = Path(user_path)
    bundled = Path(bundled_path)
    if not bundled.exists():
        return False
    if user.exists():
        raw_user = yaml.safe_load(user.read_text(encoding="utf-8")) or {}
    else:
        raw_user = {}
    raw_bundled = yaml.safe_load(bundled.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_user, dict) or not isinstance(raw_bundled, dict):
        raise ValueError("sidecar configuration must contain a YAML object")

    changed = False
    for section in ("sidecar_agents", "coprocessors"):
        source = raw_bundled.get(section) or {}
        if not isinstance(source, dict):
            continue
        target = raw_user.setdefault(section, {})
        if not isinstance(target, dict):
            raise ValueError(f"sidecar configuration section '{section}' must be a mapping")
        for item_id, definition in source.items():
            if item_id not in target:
                target[item_id] = definition
                changed = True

    if "feedback_monitor" not in raw_user and isinstance(
        raw_bundled.get("feedback_monitor"), dict
    ):
        raw_user["feedback_monitor"] = raw_bundled["feedback_monitor"]
        changed = True

    # The global bounded-call switch is policy metadata, not a worker
    # definition.  Preserve an operator's explicit value while adding the
    # bundled default to older config files that predate the switch.
    if "coprocessors_enabled" not in raw_user and "coprocessors_enabled" in raw_bundled:
        raw_user["coprocessors_enabled"] = raw_bundled["coprocessors_enabled"]
        changed = True

    if not changed:
        return False

    user.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{user.name}.", dir=str(user.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_user, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, user)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True
