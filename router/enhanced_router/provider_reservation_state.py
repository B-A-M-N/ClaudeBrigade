"""Provider-wide native-agent capacity reservation, split out of state.py.

Seventeenth increment of the incremental extraction out of ``RouteState`` --
the second slice of "Run lifecycle". This is the durable queue/admission
ledger behind the in-process request admission manager: a queued native
agent is not spawned until its ``provider_reservations`` row becomes
``reserved``. These methods are called by the (not-yet-extracted) claim/
scheduling core via ``self.``, which keeps working unchanged through the
mixin's normal method resolution order.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderReservationRepository:
    """Mixin providing provider-reservation persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def reserve_provider_agent(
        self,
        *,
        reservation_id: str,
        run_id: str,
        epoch_id: str | None,
        provider_id: str,
        execution_id: str | None,
        lane: str = "worker",
        max_active: int = 1,
        deadline_at: str | None = None,
        reason: str = "",
        enqueue: bool = True,
    ) -> dict:
        """Reserve a provider-wide native-agent slot durably.

        This complements the in-process request admission manager.  A queued
        native agent is not spawned until this record becomes ``reserved``.
        """
        if max_active < 1:
            raise ValueError("max_active must be positive")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if existing:
                conn.commit()
                return dict(existing)
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            if int(active) >= max_active and not enqueue:
                conn.rollback()
                return {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "epoch_id": epoch_id,
                    "provider_id": provider_id,
                    "execution_id": execution_id,
                    "lane": lane,
                    "state": "unavailable",
                    "reason": reason,
                }
            status = "reserved" if int(active) < max_active else "queued"
            now = _utcnow()
            conn.execute(
                "INSERT INTO provider_reservations (reservation_id, run_id, epoch_id, provider_id, execution_id,"
                " lane, state, queued_at, admitted_at, deadline_at, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, run_id, epoch_id, provider_id, execution_id, lane, status, now,
                 now if status == "reserved" else None, deadline_at, reason),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def provider_agent_capacity_available(self, provider_id: str, max_active: int) -> bool:
        """Return whether a native Agent may be spawned immediately."""
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations "
                "WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            return int(active) < max_active
        finally:
            conn.close()

    def admit_provider_agents(self, provider_id: str, max_active: int) -> list[dict]:
        conn = self._new_conn()
        admitted: list[dict] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            active = int(conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0])
            capacity = max(0, max_active - active)
            rows = conn.execute(
                "SELECT reservation_id FROM provider_reservations WHERE provider_id=? AND state='queued'"
                " ORDER BY queued_at, reservation_id LIMIT ?", (provider_id, capacity),
            ).fetchall()
            now = _utcnow()
            for row in rows:
                conn.execute(
                    "UPDATE provider_reservations SET state='reserved', admitted_at=? WHERE reservation_id=?",
                    (now, row[0]),
                )
            conn.commit()
            for row in rows:
                item = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (row[0],)).fetchone()
                if item:
                    admitted.append(dict(item))
            return admitted
        finally:
            conn.close()

    def release_provider_reservation(self, reservation_id: str, state: str = "released") -> dict | None:
        if state not in {"released", "expired", "cancelled"}:
            raise ValueError("invalid reservation terminal state")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE provider_reservations SET state=?, released_at=? WHERE reservation_id=? AND state IN ('queued','reserved')",
                (state, _utcnow(), reservation_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_provider_reservations(
        self,
        provider_id: str | None = None,
        *,
        run_id: str | None = None,
        epoch_id: str | None = None,
        active_only: bool = False,
    ) -> list[dict]:
        conn = self._new_conn()
        try:
            clauses: list[str] = []
            params: list[object] = []
            if provider_id:
                clauses.append("provider_id=?")
                params.append(provider_id)
            if run_id:
                clauses.append("run_id=?")
                params.append(run_id)
            if epoch_id:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            if active_only:
                clauses.append("state IN ('queued','reserved')")
            query = "SELECT * FROM provider_reservations"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY queued_at, reservation_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def attach_provider_reservation(self, reservation_id: str, *, execution_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE provider_reservations SET execution_id=? WHERE reservation_id=?",
                (execution_id, reservation_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def release_all_provider_reservations(self, run_id: str) -> None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE provider_reservations SET state='released', released_at=? "
                "WHERE run_id=? AND state IN ('queued','reserved')", (_utcnow(), run_id),
            )
            conn.commit()
        finally:
            conn.close()
