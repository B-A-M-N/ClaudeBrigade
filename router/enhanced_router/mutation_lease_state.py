"""Mutation lease persistence, split out of state.py.

Third increment of the incremental extraction out of ``RouteState``: these
six methods only touch the ``mutation_leases`` table through
``self._new_conn()``, so they move to their own module as a mixin without
touching any call site. ``RouteState`` still exposes these methods under
their original names.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_age(max_age_seconds: int) -> str:
    """Return an ISO-8601 timestamp that is *max_age_seconds* in the past."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    return cutoff.isoformat()


# Applied lazily inside acquire_mutation_lease itself (see its docstring) --
# there is no scheduled sweep, so a lease from a hard-killed holder (no clean
# release, no heartbeat) would otherwise block every future writer for this
# workspace indefinitely. Matches expire_stale_leases' own default.
_STALE_LEASE_MAX_AGE_SECONDS = 1_200


class MutationLeaseRepository:
    """Mixin providing mutation lease persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def acquire_mutation_lease(
        self,
        run_id: str,
        epoch_id: str,
        agent_id: str,
        role: str,
        workspace_id: str | None = None,
    ) -> bool:
        """Try to acquire a mutation lease for *agent_id*.

        Returns ``True`` on success, ``False`` if another agent already
        holds a non-released lease for this run.

        Expires a stale lease on this workspace first (heartbeat older than
        ``_STALE_LEASE_MAX_AGE_SECONDS``) before checking for an active one.
        Nothing else in production ever calls ``expire_stale_leases`` or
        ``heartbeat_mutation_lease`` on a schedule, so without this a
        hard-killed lease holder (no clean release, e.g. ``kill -9``, OOM,
        host crash -- never reaching the SubagentStop/StopFailure hook path
        that calls ``release_mutation_lease``) would otherwise lock this
        workspace for writes permanently, surviving even a router restart
        since the lease is a durable SQLite row. Same lazy-expiry-on-read
        shape as ``runnable_action_state._active_action_claims``.
        """
        lease_workspace = workspace_id or run_id
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()

            conn.execute(
                "UPDATE mutation_leases SET released_at = ? "
                "WHERE workspace_id = ? AND released_at IS NULL AND heartbeat_at < ?",
                (now, lease_workspace, _utcnow_age(_STALE_LEASE_MAX_AGE_SECONDS)),
            )

            # Check for existing active lease in this run
            active = conn.execute(
                "SELECT agent_id, workspace_id FROM mutation_leases "
                "WHERE workspace_id = ? AND released_at IS NULL",
                (lease_workspace,),
            ).fetchone()
            if active is not None:
                if active[0] == agent_id:
                    conn.execute(
                        "UPDATE mutation_leases SET heartbeat_at=? WHERE run_id=? AND agent_id=?"
                        " AND released_at IS NULL",
                        (now, run_id, agent_id),
                    )
                    conn.commit()
                    return True
                conn.rollback()
                return False

            conn.execute(
                """INSERT INTO mutation_leases (run_id, epoch_id, agent_id, role, acquired_at, heartbeat_at, workspace_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, epoch_id, agent_id, role, now, now, lease_workspace),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def release_mutation_lease(self, run_id: str, agent_id: str) -> None:
        """Release the mutation lease held by *agent_id*."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE mutation_leases SET released_at = ? WHERE run_id = ? AND agent_id = ? AND released_at IS NULL",
                (_utcnow(), run_id, agent_id),
            )
            conn.commit()
        finally:
            conn.close()

    def heartbeat_mutation_lease(self, run_id: str, agent_id: str) -> bool:
        conn = self._new_conn()
        try:
            cursor = conn.execute(
                "UPDATE mutation_leases SET heartbeat_at=? WHERE run_id=? AND agent_id=? AND released_at IS NULL",
                (_utcnow(), run_id, agent_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_mutation_lease(self, run_id: str, agent_id: str) -> dict | None:
        """Return the lease dict for *agent_id*, or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, epoch_id, agent_id, role, acquired_at, heartbeat_at, released_at, workspace_id "
                "FROM mutation_leases WHERE run_id = ? AND agent_id = ?",
                (run_id, agent_id),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "epoch_id", "agent_id", "role", "acquired_at", "heartbeat_at", "released_at", "workspace_id"), row
            ))
        finally:
            conn.close()

    def get_active_mutation_leases(
        self, run_id: str, epoch_id: str | None = None,
    ) -> list[dict]:
        """Return unreleased mutation leases for authoritative completion checks."""
        conn = self._new_conn()
        try:
            query = (
                "SELECT run_id, epoch_id, agent_id, role, acquired_at, heartbeat_at, "
                "released_at, workspace_id FROM mutation_leases "
                "WHERE run_id=? AND released_at IS NULL"
            )
            params: list[object] = [run_id]
            if epoch_id is not None:
                query += " AND epoch_id=?"
                params.append(epoch_id)
            query += " ORDER BY acquired_at, agent_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def get_active_mutator(self, run_id: str) -> dict | None:
        """Return the agent holding the unreleased lease for *run_id*, or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, epoch_id, agent_id, role, acquired_at, heartbeat_at, released_at, workspace_id "
                "FROM mutation_leases WHERE run_id = ? AND released_at IS NULL LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "epoch_id", "agent_id", "role", "acquired_at", "heartbeat_at", "released_at", "workspace_id"), row
            ))
        finally:
            conn.close()

    def expire_stale_leases(self, run_id: str, max_age_seconds: int = 1_200) -> int:
        """Release leases where *heartbeat_at* is older than *max_age_seconds*.

        Returns the number of leases released.
        """
        conn = self._new_conn()
        try:
            now = _utcnow()
            cursor = conn.execute(
                "UPDATE mutation_leases SET released_at = ? "
                "WHERE run_id = ? AND released_at IS NULL "
                "AND heartbeat_at < ?",
                (now, run_id, _utcnow_age(max_age_seconds)),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

