"""Compare-and-set role route persistence, split out of state.py.

Ninth increment of the incremental extraction out of ``RouteState``.
``RouteConflictError`` moves here too (rather than staying in state.py) so
this module is self-contained; ``state.py`` re-exports it under its
original name (``from enhanced_router.state import RouteConflictError``)
so existing importers are unaffected.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RouteConflictError(Exception):
    """Raised when a compare-and-set route update's expected version is stale."""


class CasRouteRepository:
    """Mixin providing the compare-and-set role-route update method.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def set_role_route_cas(
        self,
        run_id: str,
        epoch_id: str,
        role: str,
        model_id: str,
        command_id: str,
        expected_version: int,
        actor_type: str = "controller",
        reason: str = "",
    ) -> dict:
        """Compare-and-set route update with idempotency on command_id.

        Raises ``RouteConflictError`` when the current version does not
        match *expected_version* or the epoch is closed.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {_VALID_ROLES}")

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()

            # Verify epoch is active
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND epoch_id = ? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if not active:
                conn.rollback()
                raise ValueError("No active epoch")

            # Check idempotency: has this command_id already been applied?
            idem = conn.execute(
                "SELECT id FROM binding_commands "
                "WHERE command_id = ? AND status = 'applied' "
                "AND command_type = 'route_change'",
                (command_id,),
            ).fetchone()
            if idem is not None:
                # Return existing result
                row = conn.execute(
                    "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at "
                    "FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                    (run_id, epoch_id, role),
                ).fetchone()
                return dict(zip(
                    ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at"),
                    row,
                )) | {"idempotent": True}

            # Fetch current row so we know old model_id and version
            existing = conn.execute(
                "SELECT version, model_id FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()
            if existing is None:
                conn.rollback()
                raise ValueError(f"No existing route for {run_id}/{epoch_id}/{role}")

            current_version = existing[0]
            if current_version != expected_version:
                conn.rollback()
                raise RouteConflictError(
                    f"CAS conflict: expected version {expected_version}, "
                    f"but current version is {current_version}"
                )

            new_version = current_version + 1
            old_model_id = existing[1]

            conn.execute(
                """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                       model_id = excluded.model_id,
                       source = excluded.source,
                       reason = excluded.reason,
                       version = excluded.version,
                       changed_at = excluded.changed_at""",
                (run_id, epoch_id, role, model_id, "cas", reason, new_version, now),
            )

            conn.execute(
                "INSERT INTO route_events (run_id, epoch_id, event_type, role, old_model_id, new_model_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, "route_change", role, old_model_id, model_id, now),
            )

            # Mark command as applied
            conn.execute(
                "INSERT INTO binding_commands (command_id, run_id, epoch_id, command_type, actor_type, reason, status, applied_at, created_at) "
                "VALUES (?, ?, ?, 'route_change', ?, ?, 'applied', ?, ?)",
                (command_id, run_id, epoch_id, actor_type, reason, now, now),
            )

            conn.commit()
            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "role": role,
                "model_id": model_id,
                "source": "cas",
                "reason": reason,
                "version": new_version,
                "changed_at": now,
                "idempotent": False,
            }
        finally:
            conn.close()
