"""Run-wide resource policy and capacity accounting.

Provider admission answers "can this provider accept the request?".  This
repository answers the independent run-level question: "would admitting this
action exceed the limits assigned to this workflow?"  The checks are exposed
for planning and repeated inside the action-claim transaction so the planner
is only an optimization; the persisted claim remains authoritative.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from enhanced_router.config_models import RunResourcePolicy
from enhanced_router.repository_base import RepositoryMixin
from enhanced_router.state_errors import WorkflowStateError


_ACTIVE_EXECUTION_STATUSES = ("started", "running")
_ACTIVE_CLAIM_STATUSES = ("claimed", "consumed")
_COPROCESSOR_KINDS = frozenset({"coprocessor_call", "sidecar_call"})
_CONTROLLER_KINDS = frozenset({"controller_action", "controller_contract", "controller_integration"})
_REVIEW_ROLES = frozenset({"adversary", "reviewer", "verifier", "critical-verifier"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_json(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _policy_dict(value: object) -> dict[str, Any]:
    defaults = RunResourcePolicy().model_dump(exclude_none=False)
    raw = _parse_json(value)
    defaults.update(raw)
    # Re-validate persisted state so malformed operator edits fail closed at
    # the boundary rather than weakening a resource limit silently.
    return RunResourcePolicy(**defaults).model_dump(exclude_none=False)


class ResourcePolicyRepository(RepositoryMixin):
    """Persist and evaluate run-wide resource ceilings."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - host override
        raise NotImplementedError

    def get_run_resource_policy(self, run_id: str) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT resource_policy_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return _policy_dict(row[0] if row is not None else None)
        finally:
            conn.close()

    def set_run_resource_policy(
        self,
        run_id: str,
        policy: RunResourcePolicy | dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> dict[str, Any] | None:
        """Persist a launch-time policy without changing an active snapshot.

        A policy is normally written by ``begin_task``.  Explicit overwrite is
        available for administrative tooling but should only be used before
        work is admitted for the epoch.
        """
        parsed = (
            policy
            if isinstance(policy, RunResourcePolicy)
            else RunResourcePolicy(**policy)
        )
        encoded = json.dumps(
            parsed.model_dump(exclude_none=False), sort_keys=True, separators=(",", ":")
        )
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if overwrite:
                cursor = conn.execute(
                    "UPDATE runs SET resource_policy_json=? WHERE run_id=?",
                    (encoded, run_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE runs SET resource_policy_json=? "
                    "WHERE run_id=? AND resource_policy_json IS NULL",
                    (encoded, run_id),
                )
            if cursor.rowcount == 0:
                exists = conn.execute(
                    "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
                ).fetchone()
                if exists is None:
                    conn.rollback()
                    return None
            conn.commit()
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    @staticmethod
    def _action_item(action: dict[str, Any]) -> dict[str, Any] | None:
        kind = str(action.get("action_kind") or action.get("execution_kind") or "native_agent")
        if kind in _CONTROLLER_KINDS or bool(action.get("requires_main_controller")):
            return None
        capability = _parse_json(action.get("capability_snapshot"))
        if not capability:
            capability = _parse_json(action.get("capability_snapshot_json"))
        can_mutate = bool(action.get("can_mutate")) or bool(capability.get("can_mutate"))
        role = str(action.get("role") or "")
        is_coprocessor = kind in _COPROCESSOR_KINDS
        if is_coprocessor:
            return {
                "kind": "coprocessor",
                "can_mutate": False,
                "review": False,
                "worktree": False,
                "estimated_tokens": int(action.get("estimated_tokens") or action.get("max_output_tokens") or 0),
                "estimated_cost": float(action.get("estimated_cost") or 0.0),
            }
        return {
            "kind": "native",
            "can_mutate": can_mutate,
            "review": bool(
                capability.get("counts_as_review")
                or capability.get("may_adjudicate")
                or role in _REVIEW_ROLES
                or str(action.get("priority_class") or "") in {"review", "verification"}
            ),
            "worktree": can_mutate and str(
                action.get("workspace_policy") or capability.get("workspace_policy") or ""
            ) == "worktree",
            "estimated_tokens": int(action.get("estimated_tokens") or action.get("max_output_tokens") or 0),
            "estimated_cost": float(action.get("estimated_cost") or 0.0),
        }

    @classmethod
    def _increment_counts(cls, counts: dict[str, Any], action: dict[str, Any]) -> None:
        item = cls._action_item(action)
        if item is None:
            return
        if item["kind"] == "coprocessor":
            counts["active_coprocessors"] += 1
            counts["estimated_cost"] += float(item["estimated_cost"] or 0.0)
            return
        counts["active_native_agents"] += 1
        if item["can_mutate"]:
            counts["active_mutators"] += 1
        if item["review"]:
            counts["active_reviewers"] += 1
        if item["worktree"]:
            counts["active_worktrees"] += 1
        counts["reserved_tokens"] += int(item["estimated_tokens"] or 0)
        counts["estimated_cost"] += float(item["estimated_cost"] or 0.0)

    @classmethod
    def _capacity_reasons(
        cls,
        policy: dict[str, Any],
        counts: dict[str, Any],
        *,
        deadline_exceeded: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        limits = (
            ("active_native_agents", "max_active_native_agents", "native agent"),
            ("active_mutators", "max_active_mutators", "mutator"),
            ("active_reviewers", "max_active_reviewers", "reviewer"),
            ("active_coprocessors", "max_active_coprocessors", "coprocessor"),
            ("active_worktrees", "max_active_worktrees", "shadow worktree"),
            ("reserved_tokens", "max_reserved_tokens", "reserved tokens"),
            ("estimated_cost", "max_estimated_cost", "estimated cost"),
        )
        for count_key, limit_key, label in limits:
            limit = policy.get(limit_key)
            if limit is not None and counts[count_key] > float(limit):
                reasons.append(
                    f"run {label} limit reached ({counts[count_key]}/{limit})"
                )
        if deadline_exceeded:
            reasons.append("run deadline exceeded")
        return reasons

    @classmethod
    def _capacity_from_conn(
        cls,
        conn: sqlite3.Connection,
        run_id: str,
        epoch_id: str,
    ) -> dict[str, Any]:
        run = conn.execute(
            "SELECT created_at, resource_policy_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        policy = _policy_dict(run[1] if run is not None else None)
        counts = {
            "active_native_agents": 0,
            "active_mutators": 0,
            "active_reviewers": 0,
            "active_coprocessors": 0,
            "active_worktrees": 0,
            "reserved_tokens": 0,
            "estimated_cost": 0.0,
        }
        active_execution_ids: set[str] = set()
        rows = conn.execute(
            "SELECT execution_id, execution_kind, role, capability_snapshot_json, "
            "workspace_id, priority_class, estimated_cost FROM agent_executions "
            "WHERE run_id=? AND epoch_id=? AND status IN ('started','running')",
            (run_id, epoch_id),
        ).fetchall()
        for row in rows:
            execution_id = str(row[0])
            active_execution_ids.add(execution_id)
            action = {
                "action_kind": row[1],
                "role": row[2],
                "capability_snapshot_json": row[3],
                "workspace_policy": "worktree" if row[4] else "none",
                "priority_class": row[5],
                "estimated_cost": row[6],
            }
            cls._increment_counts(counts, action)

        spent = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost), 0) FROM agent_executions "
            "WHERE run_id=? AND epoch_id=? AND status IN "
            "('completed','failed','timeout','cancelled')",
            (run_id, epoch_id),
        ).fetchone()[0]
        counts["estimated_cost"] += float(spent or 0.0)

        # A claim without an attached live execution still reserves run
        # capacity: it may be between the controller's claim and Claude Code's
        # native lifecycle attachment.
        claim_rows = conn.execute(
            "SELECT c.action_id, c.action_kind, c.role, c.execution_id, "
            "c.intent_id, i.capability_snapshot_json, i.workspace_policy, "
            "i.priority_class FROM runnable_action_claims c "
            "LEFT JOIN spawn_intents i ON i.intent_id=c.intent_id "
            "WHERE c.run_id=? AND c.epoch_id=? AND c.status IN ('claimed','consumed')",
            (run_id, epoch_id),
        ).fetchall()
        for row in claim_rows:
            execution_id = str(row[3]) if row[3] else None
            if execution_id and execution_id in active_execution_ids:
                continue
            cls._increment_counts(
                counts,
                {
                    "action_kind": row[1],
                    "role": row[2],
                    "capability_snapshot_json": row[5],
                    "workspace_policy": row[6],
                    "priority_class": row[7],
                },
            )

        shadow_count = conn.execute(
            "SELECT COUNT(*) FROM workspaces WHERE run_id=? AND epoch_id=? "
            "AND kind='shadow' AND status='active'",
            (run_id, epoch_id),
        ).fetchone()[0]
        # Worktree reservations created by claims are not necessarily present
        # as a workspace until the native child attaches.  The count above is
        # authoritative for created worktrees; pending worktree claims were
        # counted as active_worktrees by their action item.
        counts["active_worktrees"] = max(
            int(shadow_count), counts["active_worktrees"]
        )

        reserved_tokens = conn.execute(
            "SELECT COALESCE(SUM(estimated_tokens), 0) FROM token_reservations "
            "WHERE run_id=? AND state='reserved'",
            (run_id,),
        ).fetchone()[0]
        counts["reserved_tokens"] = max(
            counts["reserved_tokens"], int(reserved_tokens or 0)
        )

        deadline_exceeded = False
        deadline_seconds = policy.get("deadline_seconds")
        if deadline_seconds is not None and run is not None and run[0]:
            try:
                created = datetime.fromisoformat(str(run[0]))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                deadline_exceeded = _utcnow() >= created + timedelta(seconds=float(deadline_seconds))
            except (TypeError, ValueError, OverflowError):
                deadline_exceeded = True
        reasons = cls._capacity_reasons(
            policy, counts, deadline_exceeded=deadline_exceeded
        )
        remaining: dict[str, int | float | None] = {}
        for count_key, limit_key in (
            ("active_native_agents", "max_active_native_agents"),
            ("active_mutators", "max_active_mutators"),
            ("active_reviewers", "max_active_reviewers"),
            ("active_coprocessors", "max_active_coprocessors"),
            ("active_worktrees", "max_active_worktrees"),
            ("reserved_tokens", "max_reserved_tokens"),
            ("estimated_cost", "max_estimated_cost"),
        ):
            limit = policy.get(limit_key)
            if limit is None:
                remaining[count_key] = None
            elif count_key == "estimated_cost":
                remaining[count_key] = max(0.0, float(limit) - float(counts[count_key]))
            else:
                remaining[count_key] = max(0, int(limit) - int(counts[count_key]))
        return {
            "policy": policy,
            "active": counts,
            "remaining": remaining,
            "blocked": reasons,
            "deadline_exceeded": deadline_exceeded,
        }

    def get_run_resource_capacity(self, run_id: str, epoch_id: str) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            return self._capacity_from_conn(conn, run_id, epoch_id)
        finally:
            conn.close()

    def _assert_action_capacity(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        epoch_id: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = self._capacity_from_conn(conn, run_id, epoch_id)
        projected = dict(snapshot["active"])
        self._increment_counts(projected, action)
        reasons = self._capacity_reasons(
            snapshot["policy"], projected,
            deadline_exceeded=bool(snapshot.get("deadline_exceeded")),
        )
        if reasons:
            raise WorkflowStateError("; ".join(reasons))
        return snapshot

    def assert_run_resource_capacity(
        self, run_id: str, epoch_id: str, action: dict[str, Any]
    ) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            return self._assert_action_capacity(conn, run_id, epoch_id, action)
        finally:
            conn.close()
