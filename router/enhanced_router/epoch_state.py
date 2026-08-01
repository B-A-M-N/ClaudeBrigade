"""Epoch lifecycle persistence, split out of state.py.

Seventh increment of the incremental extraction out of ``RouteState``. Epoch
close cascades cleanup across several tables owned by other repositories
(bindings, mutation leases, provider reservations, executions) via direct
SQL rather than calling their methods, so it moves as a self-contained unit
without creating any cross-module method dependency. ``RouteState`` still
exposes these methods under their original names; other sections that call
``self.get_active_epoch(...)`` keep working unchanged via the mixin's normal
method resolution order.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpochRepository:
    """Mixin providing epoch lifecycle persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def get_active_epoch(self, run_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * "
                "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def create_epoch(self, run_id: str, epoch_id: str, workflow_id: str, profile_id: str | None = None) -> dict:
        """Create epoch row. Raises ValueError if active epoch already exists."""
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                (run_id,),
            ).fetchone()
            if active:
                raise ValueError(f"Active epoch already exists for run {run_id}")

            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                (run_id, epoch_id, workflow_id, profile_id, _utcnow()),
            )
            conn.commit()
            return self.get_active_epoch(run_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def close_epoch(self, run_id: str, epoch_id: str) -> None:
        """Set status='closed' and closed_at. Release orphaned bindings."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE epochs SET status = 'closed', closed_at = ? WHERE run_id = ? AND epoch_id = ?",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE agent_bindings SET released_at = ? WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE controller_bindings SET released_at=? WHERE run_id=? AND released_at IS NULL",
                (_utcnow(), run_id),
            )
            conn.execute(
                "UPDATE provider_reservations SET state='released', released_at=? "
                "WHERE run_id=? AND epoch_id=? AND state IN ('queued','reserved')",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE mutation_leases SET released_at=? WHERE run_id=? AND epoch_id=? AND released_at IS NULL",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE agent_executions SET status='cancelled', completed_at=?, updated_at=? "
                "WHERE run_id=? AND epoch_id=? AND status IN ('started','running')",
                (_utcnow(), _utcnow(), run_id, epoch_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_profile_routes_atomic(
        self, run_id: str, epoch_id: str, profile_id: str, reason: str
    ) -> dict[str, dict]:
        """BEGIN IMMEDIATE transaction: set all 4 role routes from profile.

        Updates epochs.profile_id, upserts each role route, appends a
        *route_event* row per changed role, and returns the actual persisted
        versions.  One failure rolls back ALL changes.
        """

        from enhanced_router.registry import get_registry

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                now = _utcnow()
                reg = get_registry()
                reg.load_profiles()
                profile = reg.get_profile(profile_id)

                # 1. Update epochs.profile_id on the active epoch row
                conn.execute(
                    "UPDATE epochs SET profile_id = ? WHERE run_id = ? AND epoch_id = ? AND closed_at IS NULL",
                    (profile_id, run_id, epoch_id),
                )

                results: dict[str, dict] = {}
                for role in ("recon", "implementer", "adversary", "repairer"):
                    target = profile.route_target(role)
                    model_id = target.model
                    conn.execute(
                        """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                           VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                           ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                               model_id = excluded.model_id,
                               source = excluded.source,
                               reason = excluded.reason,
                               version = version + 1,
                               changed_at = excluded.changed_at""",
                        (run_id, epoch_id, role, model_id, reason, now),
                    )
                    conn.execute(
                        "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                    )
                    conn.execute(
                        "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (json.dumps(target.fallback_models), run_id, epoch_id, role),
                    )

                    # 2. Read actual version after upsert
                    row = conn.execute(
                        "SELECT version, model_id FROM role_routes WHERE run_id=? AND epoch_id=? AND role=?",
                        (run_id, epoch_id, role),
                    ).fetchone()
                    actual_version = row[0] if row else 1

                    # 3. Append route_event for this role
                    conn.execute(
                        "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                        "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                        (run_id, epoch_id, role, model_id, now),
                    )

                    results[role] = {
                        "run_id": run_id,
                        "epoch_id": epoch_id,
                        "role": role,
                        "model_id": model_id,
                        "source": "profile",
                        "reason": reason,
                        "version": actual_version,
                    }

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return results
        finally:
            conn.close()

    def create_epoch_from_profile(
        self, run_id: str, epoch_id: str, workflow_id: str, profile_id: str
    ) -> dict:
        """Transaction: create_epoch + set profile routes. Returns epoch dict.

        Routes are NOT optional -- a profile load failure propagates as an
        exception so the caller knows the epoch has no routes.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Check for active epoch
                active = conn.execute(
                    "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                    (run_id,),
                ).fetchone()
                if active:
                    conn.rollback()
                    raise ValueError(
                        f"Active epoch already exists for run {run_id}"
                    )

                # Create the epoch row
                conn.execute(
                    "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                    (run_id, epoch_id, workflow_id, profile_id, _utcnow()),
                )

                # Load profile and set routes -- raise on failure
                from enhanced_router.registry import get_registry

                reg = get_registry()
                reg.load_profiles()
                profile = reg.get_profile(profile_id)
                now = _utcnow()
                for role in ("recon", "implementer", "adversary", "repairer"):
                    target = profile.route_target(role)
                    model_id = target.model
                    conn.execute(
                        """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                           VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                           ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                               model_id = excluded.model_id,
                               source = excluded.source,
                               reason = excluded.reason,
                               version = version + 1,
                               changed_at = excluded.changed_at""",
                        (run_id, epoch_id, role, model_id, f"profile:{profile_id}", now),
                    )
                    conn.execute(
                        "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                    )
                    conn.execute(
                        "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (json.dumps(target.fallback_models), run_id, epoch_id, role),
                    )

                    # Append route_event for each role
                    conn.execute(
                        "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                        "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                        (run_id, epoch_id, role, model_id, now),
                    )

                conn.commit()
                epoch = conn.execute(
                    "SELECT id, run_id, epoch_id, workflow_id, profile_id, status, created_at, closed_at "
                    "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                    (run_id,),
                ).fetchone()
                if epoch is None:
                    raise RuntimeError(
                        f"Failed to create epoch {epoch_id} for run {run_id}"
                    )
                return dict(
                    zip(
                        (
                            "id",
                            "run_id",
                            "epoch_id",
                            "workflow_id",
                            "profile_id",
                            "status",
                            "created_at",
                            "closed_at",
                        ),
                        epoch,
                    )
                )
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()
