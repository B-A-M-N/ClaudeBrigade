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

from enhanced_router.repository_base import RepositoryMixin

import sqlite3
from datetime import datetime, timezone

from enhanced_router.model_health_state import evaluate_model_health


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderReservationRepository(RepositoryMixin):
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
        model_id: str | None = None,
        health_max_age_seconds: float | None = None,
        allow_untested_models: bool | None = None,
        lane_limit: int | None = None,
    ) -> dict:
        """Reserve a provider-wide native-agent slot durably.

        This complements the in-process request admission manager.  A queued
        native agent is not spawned until this record becomes ``reserved``.

        When *model_id* is given, the latest model-health record is checked at
        both queue time and promotion time.  A model with no health record is
        admitted only when *allow_untested_models* permits it.  Stale or
        failed records never remain silently queued until a provider slot
        happens to become available.
        """
        if max_active < 1:
            raise ValueError("max_active must be positive")
        health_max_age_seconds, allow_untested_models = self._provider_health_policy(
            provider_id,
            health_max_age_seconds=health_max_age_seconds,
            allow_untested_models=allow_untested_models,
        )
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if existing:
                conn.commit()
                return dict(existing)
            if model_id is not None:
                health_row = conn.execute(
                    "SELECT status, reachable, authenticated, compatible, checked_at "
                    "FROM model_health WHERE model_id=? ORDER BY checked_at DESC LIMIT 1",
                    (model_id,),
                ).fetchone()
                health = evaluate_model_health(
                    health_row,
                    max_age_seconds=health_max_age_seconds,
                    allow_untested=allow_untested_models,
                )
                if not bool(health["admissible"]):
                    conn.rollback()
                    return {
                        "reservation_id": reservation_id,
                        "run_id": run_id,
                        "epoch_id": epoch_id,
                        "provider_id": provider_id,
                        "execution_id": execution_id,
                        "lane": lane,
                        "model_id": model_id,
                        "state": "unavailable",
                        "reason": f"model {model_id!r} is {health['status']}: {health['reason']}",
                    }
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            lane_active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations "
                "WHERE provider_id=? AND lane=? AND state='reserved'",
                (provider_id, lane),
            ).fetchone()[0]
            if lane_limit is not None and int(lane_active) >= lane_limit and not enqueue:
                conn.rollback()
                return {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "epoch_id": epoch_id,
                    "provider_id": provider_id,
                    "execution_id": execution_id,
                    "lane": lane,
                    "model_id": model_id,
                    "state": "unavailable",
                    "reason": reason,
                }
            if int(active) >= max_active and not enqueue:
                conn.rollback()
                return {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "epoch_id": epoch_id,
                    "provider_id": provider_id,
                    "execution_id": execution_id,
                    "lane": lane,
                    "model_id": model_id,
                    "state": "unavailable",
                    "reason": reason,
                }
            status = "reserved" if int(active) < max_active else "queued"
            now = _utcnow()
            conn.execute(
                "INSERT INTO provider_reservations (reservation_id, run_id, epoch_id, provider_id, execution_id,"
                " lane, state, queued_at, admitted_at, deadline_at, reason, model_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, run_id, epoch_id, provider_id, execution_id, lane, status, now,
                 now if status == "reserved" else None, deadline_at, reason, model_id),
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

    def provider_agent_capacity_available(
        self,
        provider_id: str,
        max_active: int,
        lane: str = "worker",
        lane_limit: int | None = None,
    ) -> bool:
        """Return whether a native Agent may be spawned immediately."""
        return self.provider_agent_capacity_remaining(
            provider_id,
            max_active,
            lane=lane,
            lane_limit=lane_limit,
        ) > 0

    def provider_agent_capacity_remaining(
        self,
        provider_id: str,
        max_active: int,
        lane: str = "worker",
        lane_limit: int | None = None,
    ) -> int:
        """Return immediate capacity for a provider/lane.

        Runnable-action waves use this to avoid advertising more work than a
        provider can admit.  Claims still perform the authoritative atomic
        reservation, so this helper is an optimization for scheduling and
        never replaces claim-time admission.
        """
        if max_active < 1:
            return 0
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations "
                "WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            remaining = max(0, int(max_active) - int(active))
            if lane_limit is None:
                return remaining
            lane_active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations "
                "WHERE provider_id=? AND lane=? AND state='reserved'",
                (provider_id, lane),
            ).fetchone()[0]
            return max(0, min(remaining, int(lane_limit) - int(lane_active)))
        finally:
            conn.close()

    def admit_provider_agents(
        self,
        provider_id: str,
        max_active: int,
        *,
        health_max_age_seconds: float | None = None,
        allow_untested_models: bool | None = None,
    ) -> list[dict]:
        """Promote queued reservations whose model route is still admissible.

        Health is checked in the same ``BEGIN IMMEDIATE`` transaction that
        promotes the reservation.  A stale/unhealthy candidate is terminally
        expired and skipped, allowing a later healthy candidate to use the
        newly available slot without being blocked by FIFO poison.
        """
        health_max_age_seconds, allow_untested_models = self._provider_health_policy(
            provider_id,
            health_max_age_seconds=health_max_age_seconds,
            allow_untested_models=allow_untested_models,
        )
        conn = self._new_conn()
        admitted_ids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            active = int(conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0])
            capacity = max(0, max_active - active)
            rows = conn.execute(
                "SELECT reservation_id, model_id FROM provider_reservations "
                "WHERE provider_id=? AND state='queued' "
                "ORDER BY queued_at, reservation_id", (provider_id,),
            ).fetchall()
            now = _utcnow()
            for row in rows:
                if len(admitted_ids) >= capacity:
                    break
                model_id = row[1]
                if model_id:
                    health_row = conn.execute(
                        "SELECT status, reachable, authenticated, compatible, checked_at "
                        "FROM model_health WHERE model_id=? ORDER BY checked_at DESC LIMIT 1",
                        (model_id,),
                    ).fetchone()
                    health = evaluate_model_health(
                        health_row,
                        max_age_seconds=health_max_age_seconds,
                        allow_untested=allow_untested_models,
                    )
                    if not bool(health["admissible"]):
                        conn.execute(
                            "UPDATE provider_reservations SET state='expired', released_at=?, "
                            "reason=? WHERE reservation_id=? AND state='queued'",
                            (now, f"model {model_id!r} is {health['status']}: {health['reason']}", row[0]),
                        )
                        continue
                conn.execute(
                    "UPDATE provider_reservations SET state='reserved', admitted_at=? WHERE reservation_id=?",
                    (now, row[0]),
                )
                admitted_ids.append(str(row[0]))
            conn.commit()
            result_rows = []
            for reservation_id in admitted_ids:
                item = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
                if item:
                    result_rows.append(dict(item))
            return result_rows
        finally:
            conn.close()

    def _provider_health_policy(
        self,
        provider_id: str,
        *,
        health_max_age_seconds: float | None,
        allow_untested_models: bool | None,
    ) -> tuple[float, bool]:
        """Resolve provider health policy without making registry admission mandatory."""
        if health_max_age_seconds is None or allow_untested_models is None:
            try:
                from enhanced_router.registry import get_registry

                provider = get_registry().providers.get(provider_id)
            except Exception:
                provider = None
            if provider is not None:
                if health_max_age_seconds is None:
                    health_max_age_seconds = float(
                        getattr(provider, "health_max_age_seconds", 900.0)
                    )
                if allow_untested_models is None:
                    allow_untested_models = bool(
                        getattr(provider, "allow_untested_models", True)
                    )
        return (
            float(health_max_age_seconds if health_max_age_seconds is not None else 900.0),
            bool(True if allow_untested_models is None else allow_untested_models),
        )

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
