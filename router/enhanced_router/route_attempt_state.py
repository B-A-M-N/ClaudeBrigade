"""Durable telemetry for each concrete route-ladder candidate attempt."""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RouteAttemptRepository(RepositoryMixin):
    """Persist bounded coprocessor/provider candidate attempts."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover
        raise NotImplementedError

    def start_route_attempt(
        self, *, attempt_id: str, run_id: str, epoch_id: str, execution_id: str,
        candidate_index: int, model_id: str, provider_id: str | None,
        endpoint_id: str, route_digest: str,
    ) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            now = _utcnow()
            conn.execute(
                "INSERT INTO route_attempts "
                "(attempt_id, run_id, epoch_id, execution_id, candidate_index, model_id, "
                "provider_id, endpoint_id, route_digest, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'started', ?)",
                (attempt_id, run_id, epoch_id, execution_id, candidate_index, model_id,
                 provider_id, endpoint_id, route_digest, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM route_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def finish_route_attempt(
        self, attempt_id: str, *, status: str, status_code: int | None = None,
        error_class: str | None = None, error: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if status not in {"succeeded", "failed", "cancelled", "skipped"}:
            raise ValueError(f"invalid route attempt status: {status}")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE route_attempts SET status=?, status_code=?, error_class=?, "
                "error=?, usage_json=?, finished_at=? WHERE attempt_id=?",
                (status, status_code, error_class, (error or "")[:500],
                 json.dumps(usage or {}, separators=(",", ":")), _utcnow(), attempt_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM route_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_route_attempts(
        self, run_id: str, epoch_id: str, execution_id: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            if execution_id is None:
                rows = conn.execute(
                    "SELECT * FROM route_attempts WHERE run_id=? AND epoch_id=? "
                    "ORDER BY started_at, candidate_index", (run_id, epoch_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM route_attempts WHERE run_id=? AND epoch_id=? "
                    "AND execution_id=? ORDER BY candidate_index, started_at",
                    (run_id, epoch_id, execution_id),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def reconcile_route_attempts(
        self,
        execution_ids: list[str] | tuple[str, ...] | set[str],
        *,
        error_class: str = "orphaned_after_restart",
        error: str = "route attempt was interrupted during router recovery",
    ) -> int:
        """Close route attempts left in ``started`` by a crashed process.

        An attempt is deliberately marked failed rather than succeeded: a
        detached fastpath request has no durable proof that its provider
        response reached the router.  This keeps route-ladder telemetry and
        retry decisions honest after a restart.
        """
        ids = [str(item) for item in execution_ids if str(item)]
        if not ids:
            return 0
        conn = self._new_conn()
        try:
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.execute(
                "UPDATE route_attempts SET status='failed', error_class=?, error=?, "
                "finished_at=? WHERE status='started' AND execution_id IN ("
                + placeholders + ")",
                [error_class, error[:500], _utcnow(), *ids],
            )
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()
