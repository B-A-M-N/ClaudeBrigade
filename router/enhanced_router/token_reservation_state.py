"""Durable per-run token budget reservations."""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timedelta, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenReservationRepository(RepositoryMixin):
    """Reserve estimated work before an action can be spawned."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover
        raise NotImplementedError

    def reserve_token_budget(
        self,
        *,
        reservation_id: str,
        run_id: str,
        epoch_id: str | None,
        action_id: str | None,
        execution_id: str | None,
        estimated_tokens: int,
    ) -> dict:
        if estimated_tokens < 1:
            raise ValueError("estimated_tokens must be positive")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM token_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return dict(existing)
            run = conn.execute(
                "SELECT token_budget, resource_policy_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            budget = run[0] if run is not None else None
            resource_policy: dict = {}
            if run is not None and run[1]:
                try:
                    parsed = json.loads(run[1])
                    if isinstance(parsed, dict):
                        resource_policy = parsed
                except (TypeError, ValueError, json.JSONDecodeError):
                    conn.rollback()
                    return {
                        "reservation_id": reservation_id,
                        "run_id": run_id,
                        "state": "unavailable",
                        "reason": "run resource policy is malformed",
                    }
            if budget is not None:
                spent = conn.execute(
                    "SELECT COALESCE(SUM(total_tokens), 0) FROM agent_executions WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
                reserved = conn.execute(
                    "SELECT COALESCE(SUM(estimated_tokens), 0) FROM token_reservations "
                    "WHERE run_id=? AND state='reserved'",
                    (run_id,),
                ).fetchone()[0]
                if int(spent or 0) + int(reserved or 0) + estimated_tokens > int(budget):
                    conn.rollback()
                    return {
                        "reservation_id": reservation_id,
                        "run_id": run_id,
                        "state": "unavailable",
                        "reason": (
                            f"run {run_id!r} has exhausted its token budget: "
                            f"spent={spent}, reserved={reserved}, "
                            f"requested={estimated_tokens}, budget={budget}"
                        ),
                    }
            max_reserved = resource_policy.get("max_reserved_tokens")
            if max_reserved is not None:
                reserved = conn.execute(
                    "SELECT COALESCE(SUM(estimated_tokens), 0) FROM token_reservations "
                    "WHERE run_id=? AND state='reserved'",
                    (run_id,),
                ).fetchone()[0]
                if int(reserved or 0) + estimated_tokens > int(max_reserved):
                    conn.rollback()
                    return {
                        "reservation_id": reservation_id,
                        "run_id": run_id,
                        "state": "unavailable",
                        "reason": (
                            f"run {run_id!r} has exhausted its reserved-token limit: "
                            f"reserved={reserved}, requested={estimated_tokens}, "
                            f"limit={max_reserved}"
                        ),
                    }
            now = _utcnow()
            conn.execute(
                "INSERT INTO token_reservations "
                "(reservation_id, run_id, epoch_id, action_id, execution_id, estimated_tokens, state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?)",
                (reservation_id, run_id, epoch_id, action_id, execution_id, estimated_tokens, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM token_reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def release_token_reservation(self, reservation_id: str, state: str = "released") -> dict | None:
        if state not in {"consumed", "released", "expired", "cancelled"}:
            raise ValueError("invalid token reservation state")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE token_reservations SET state=?, released_at=? "
                "WHERE reservation_id=? AND state='reserved'",
                (state, _utcnow(), reservation_id),
            )
            row = conn.execute(
                "SELECT * FROM token_reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            conn.commit()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_token_reservations(
        self,
        run_id: str | None = None,
        *,
        epoch_id: str | None = None,
        active_only: bool = False,
    ) -> list[dict]:
        """Return token reservations for presentation and reconciliation.

        Token reservations are intentionally exposed as a read-only ledger
        projection.  Callers must use ``reserve_token_budget`` and
        ``release_token_reservation`` for state transitions.
        """
        conn = self._new_conn()
        try:
            clauses: list[str] = []
            params: list[object] = []
            if run_id is not None:
                clauses.append("run_id=?")
                params.append(run_id)
            if epoch_id is not None:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            if active_only:
                clauses.append("state='reserved'")
            query = "SELECT * FROM token_reservations"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at, reservation_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def reconcile_token_reservations(
        self,
        run_id: str | None = None,
        *,
        max_age_seconds: int = 900,
    ) -> int:
        """Release only reservations proven to be orphaned.

        A router restart must not invalidate live controller or worker
        reservations. Claims and executions are the authoritative liveness
        records; an unlinked reservation is expired only after the crash
        recovery age has elapsed.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            clauses = ["tr.state='reserved'"]
            params: list[object] = []
            if run_id is not None:
                clauses.append("tr.run_id=?")
                params.append(run_id)
            cutoff = (
                datetime.now(timezone.utc) - timedelta(seconds=max(1, max_age_seconds))
            ).isoformat()
            clauses.append(
                "((c.action_id IS NOT NULL AND c.status NOT IN ('claimed','consumed')) "
                "OR (tr.execution_id IS NOT NULL AND e.execution_id IS NOT NULL "
                "AND e.status IN ('completed','failed','timeout','cancelled')) "
                "OR (c.action_id IS NULL AND e.execution_id IS NULL AND tr.created_at < ?))"
            )
            params.append(cutoff)
            rows = conn.execute(
                "SELECT tr.reservation_id FROM token_reservations AS tr "
                "LEFT JOIN runnable_action_claims AS c "
                "ON c.token_reservation_id=tr.reservation_id "
                "LEFT JOIN agent_executions AS e ON e.execution_id=tr.execution_id "
                "WHERE " + " AND ".join(clauses),
                params,
            ).fetchall()
            if not rows:
                conn.commit()
                return 0
            reservation_ids = [str(row[0]) for row in rows]
            conn.execute(
                "UPDATE token_reservations SET state='expired', released_at=? "
                "WHERE state='reserved' AND reservation_id IN ("
                + ",".join("?" for _ in reservation_ids) + ")",
                [_utcnow(), *reservation_ids],
            )
            conn.commit()
            return len(reservation_ids)
        finally:
            conn.close()
