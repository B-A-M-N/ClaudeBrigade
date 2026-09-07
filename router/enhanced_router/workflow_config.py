"""Non-destructive migration for the persisted workflow configuration."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_SCHEMA_VERSION = 2


def _merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> bool:
    changed = False
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
            changed = True
        elif isinstance(target[key], dict) and isinstance(value, dict):
            changed = _merge_missing(target[key], value) or changed
    return changed


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked configuration: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _phase_map(phases: object) -> dict[str, dict[str, Any]]:
    if not isinstance(phases, list):
        return {}
    return {
        str(item.get("id")): item
        for item in phases
        if isinstance(item, dict) and item.get("id")
    }


def migrate_workflow_config(
    user_path: str | Path,
    bundled_path: str | Path,
) -> bool:
    """Upgrade old workflows without replacing operator-owned decisions.

    A legacy workflow whose phase IDs are a subset of the bundled default is
    recognized as the old default shape.  It receives the bundled phase graph
    (including package planning and all-package completion), with every
    existing phase value winning over the bundled default.  Workflows with
    custom phase IDs are not restructured; only missing fields on matching
    phases and resource-policy keys are filled.
    """
    user = Path(user_path)
    bundled = Path(bundled_path)
    if not bundled.exists():
        return False
    raw_user = yaml.safe_load(user.read_text(encoding="utf-8")) if user.exists() else {}
    raw_bundled = yaml.safe_load(bundled.read_text(encoding="utf-8")) or {}
    if raw_user is None:
        raw_user = {}
    if not isinstance(raw_user, dict) or not isinstance(raw_bundled, dict):
        raise ValueError("workflow configuration must contain a YAML object")

    user_workflows = raw_user.setdefault("workflows", {})
    bundled_workflows = raw_bundled.get("workflows") or {}
    if not isinstance(user_workflows, dict) or not isinstance(bundled_workflows, dict):
        raise ValueError("workflow configuration 'workflows' must be a mapping")

    changed = False
    for workflow_id, bundled_workflow in bundled_workflows.items():
        if not isinstance(bundled_workflow, dict):
            continue
        user_workflow = user_workflows.get(workflow_id)
        if not isinstance(user_workflow, dict):
            continue

        if "default_profile" not in user_workflow and "default_profile" in bundled_workflow:
            user_workflow["default_profile"] = bundled_workflow["default_profile"]
            changed = True
        if isinstance(user_workflow.get("resource_policy"), dict) and isinstance(
            bundled_workflow.get("resource_policy"), dict
        ):
            changed = _merge_missing(
                user_workflow["resource_policy"], bundled_workflow["resource_policy"]
            ) or changed
        elif "resource_policy" not in user_workflow and "resource_policy" in bundled_workflow:
            user_workflow["resource_policy"] = copy.deepcopy(bundled_workflow["resource_policy"])
            changed = True

        user_phases = user_workflow.get("phases")
        bundled_phases = bundled_workflow.get("phases")
        bundled_phase_list = bundled_phases if isinstance(bundled_phases, list) else []
        user_by_id = _phase_map(user_phases)
        bundled_by_id = _phase_map(bundled_phases)
        if not user_by_id or not bundled_by_id:
            continue
        user_ids = set(user_by_id)
        bundled_ids = set(bundled_by_id)
        if user_ids <= bundled_ids:
            # This is the old bundled default shape.  Preserve edits in each
            # existing phase while adding newly required phases in canonical
            # workflow order.
            migrated: list[dict[str, Any]] = []
            for bundled_phase in bundled_phase_list:
                phase_id = str(bundled_phase.get("id"))
                merged = copy.deepcopy(bundled_phase)
                if phase_id in user_by_id:
                    merged.update(copy.deepcopy(user_by_id[phase_id]))
                if (
                    merged.get("fanout_from") == "work_packages"
                    and "package-plan" in bundled_by_id
                ):
                    dependencies = [
                        str(item) for item in (merged.get("depends_on") or [])
                    ]
                    if "package-plan" not in dependencies:
                        # Keep legacy prerequisites, but never allow a
                        # package-fanout worker to bypass the controller's
                        # persisted package plan.
                        merged["depends_on"] = [*dependencies, "package-plan"]
                migrated.append(merged)
            if migrated != user_phases:
                user_workflow["phases"] = migrated
                changed = True
        else:
            # Custom workflows retain their phase graph.  They still receive
            # new fields on any phase that has a bundled counterpart.
            for phase_id, user_phase in user_by_id.items():
                bundled_phase = bundled_by_id.get(phase_id)
                if bundled_phase is not None:
                    changed = _merge_missing(user_phase, bundled_phase) or changed

    if raw_user.get("workflow_schema_version") != WORKFLOW_SCHEMA_VERSION:
        raw_user["workflow_schema_version"] = WORKFLOW_SCHEMA_VERSION
        changed = True
    if not changed:
        return False
    _atomic_write(user, raw_user)
    return True
