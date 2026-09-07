"""Binding command persistence, split out of state.py.

Eleventh increment of the incremental extraction out of ``RouteState``.
``VALID_COMMAND_TYPES`` moves here too (rather than staying in state.py) so
this module is self-contained; ``state.py`` re-exports it under its
original name (``from enhanced_router.state import VALID_COMMAND_TYPES``)
so existing importers (``mcp_control.py``, tests) are unaffected.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import sqlite3
from datetime import datetime, timezone

VALID_COMMAND_TYPES = frozenset((
    "model_change", "profile_set", "route_change",
    "binding_release", "binding_reenable",
))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BindingCommandRepository(RepositoryMixin):
    """Mixin providing binding command persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def record_binding_command(
        self,
        command_id: str,
        run_id: str,
        epoch_id: str,
        command_type: str,
        actor_type: str = "controller",
        reason: str = "",
        claude_session_id: str = "",
        claude_agent_id: str = "",
        expected_binding_version: int | None = None,
        requested_model_id: str | None = None,
        requested_role: str | None = None,
    ) -> dict:
        """Insert a new binding command in *pending* status."""
        if command_type not in VALID_COMMAND_TYPES:
            raise ValueError(f"Invalid command_type: {command_type}. Must be one of {VALID_COMMAND_TYPES}")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            conn.execute(
                """INSERT INTO binding_commands
                   (command_id, run_id, epoch_id, claude_session_id, command_type,
                    actor_type, reason, claude_agent_id, expected_binding_version,
                    requested_model_id, requested_role, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    command_id, run_id, epoch_id, claude_session_id, command_type,
                    actor_type, reason, claude_agent_id, expected_binding_version,
                    requested_model_id, requested_role, now,
                ),
            )
            conn.commit()
            return self.get_binding_command_by_id(command_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_binding_command_by_id(self, command_id: str) -> dict | None:
        """Return a command dict by its command_id."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT id, command_id, run_id, epoch_id, claude_session_id, command_type, "
                "actor_type, reason, claude_agent_id, expected_binding_version, "
                "requested_model_id, requested_role, status, applied_at, actor_id, created_at "
                "FROM binding_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "command_id", "run_id", "epoch_id", "claude_session_id", "command_type",
                 "actor_type", "reason", "claude_agent_id", "expected_binding_version",
                 "requested_model_id", "requested_role", "status", "applied_at", "actor_id", "created_at"),
                row,
            ))
        finally:
            conn.close()

    def get_binding_commands(
        self, run_id: str, epoch_id: str, limit: int = 50
    ) -> list[dict]:
        """Return commands for a run/epoch ordered newest first."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT id, command_id, run_id, epoch_id, claude_session_id, command_type, "
                "actor_type, reason, claude_agent_id, expected_binding_version, "
                "requested_model_id, requested_role, status, applied_at, actor_id, created_at "
                "FROM binding_commands WHERE run_id = ? AND epoch_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (run_id, epoch_id, limit),
            ).fetchall()
            return [
                dict(zip(
                    ("id", "command_id", "run_id", "epoch_id", "claude_session_id", "command_type",
                     "actor_type", "reason", "claude_agent_id", "expected_binding_version",
                     "requested_model_id", "requested_role", "status", "applied_at", "actor_id", "created_at"),
                    row,
                ))
                for row in rows
            ]
        finally:
            conn.close()

    def apply_binding_command(
        self, command_id: str, actor_type: str = "controller", actor_id: str = ""
    ) -> dict:
        """Change a pending command to *applied* status."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id, status FROM binding_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if existing is None:
                conn.rollback()
                raise ValueError(f"Command '{command_id}' not found")
            if existing[1] != "pending":
                conn.rollback()
                raise ValueError(f"Command '{command_id}' is not pending (status: {existing[1]})")
            now = _utcnow()
            conn.execute(
                "UPDATE binding_commands SET status = 'applied', applied_at = ?, actor_type = ?, actor_id = ? "
                "WHERE command_id = ?",
                (now, actor_type, actor_id, command_id),
            )
            conn.commit()
            return self.get_binding_command_by_id(command_id)  # type: ignore[return-value]
        finally:
            conn.close()
