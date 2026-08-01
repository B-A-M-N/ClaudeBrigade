"""LiteLLM generation/deployment persistence, split out of state.py.

First increment of an incremental extraction out of ``RouteState`` (a single
~7600-line class): the generation/deployment bookkeeping behind blue-green
LiteLLM catalog replacement is self-contained -- every method here only
touches the ``litellm_generations``/``litellm_deployments`` tables through
``self._new_conn()`` -- so it can move to its own module as a mixin without
touching any call site. ``RouteState`` still exposes these methods under
their original names; nothing outside this file changes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiteLLMGenerationRepository:
    """Mixin providing LiteLLM generation/deployment lifecycle methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def create_litellm_generation(
        self,
        registry_hash: str,
        model_count: int,
        config_digest: str,
        reason: str = "",
    ) -> int:
        """Create a new staging generation. Returns generation number."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            cursor = conn.execute(
                "INSERT INTO litellm_generations (registry_hash, model_count, config_digest, status, reason, created_at) "
                "VALUES (?, ?, ?, 'staging', ?, ?)",
                (registry_hash, model_count, config_digest, reason, now),
            )
            assert cursor.lastrowid is not None
            generation: int = int(cursor.lastrowid)
            conn.commit()
            return generation
        finally:
            conn.close()

    def activate_litellm_generation(self, generation: int) -> None:
        """Transition a generation from staging to active.
        Retires the previously active generation.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            # Retire current active
            conn.execute(
                "UPDATE litellm_generations SET status = 'retired', retired_at = ? "
                "WHERE status = 'active'",
                (now,),
            )
            # Activate the new generation
            conn.execute(
                "UPDATE litellm_generations SET status = 'active', activated_at = ? "
                "WHERE generation = ?",
                (now, generation),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active_litellm_generation(self) -> dict | None:
        """Return the currently active generation row or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT generation, registry_hash, model_count, config_digest, status, reason, "
                "created_at, activated_at, retired_at "
                "FROM litellm_generations WHERE status = 'active' LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("generation", "registry_hash", "model_count", "config_digest", "status",
                 "reason", "created_at", "activated_at", "retired_at"),
                row,
            ))
        finally:
            conn.close()

    def register_litellm_deployment(
        self, generation: int, port: int, pid: int | None = None
    ) -> int:
        """Record a deployment instance for a generation. Returns deployment id."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            cursor = conn.execute(
                "INSERT INTO litellm_deployments (generation, port, pid, status, created_at) "
                "VALUES (?, ?, ?, 'starting', ?)",
                (generation, port, pid, now),
            )
            assert cursor.lastrowid is not None
            dep_id: int = int(cursor.lastrowid)
            conn.commit()
            return dep_id
        finally:
            conn.close()

    def update_litellm_deployment(
        self,
        dep_id: int,
        status: str | None = None,
        pid: int | None = None,
        termination_reason: str | None = None,
    ) -> None:
        """Update deployment status and/or pid and termination reason."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            sets = ["health_checked_at = ?"]
            params: list = [now]
            if status is not None:
                sets.append("status = ?")
                params.append(status)
            if pid is not None:
                sets.append("pid = ?")
                params.append(pid)
            if termination_reason is not None:
                sets.append("termination_reason = ?")
                params.append(termination_reason)
            params.append(dep_id)
            conn.execute(
                f"UPDATE litellm_deployments SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()

    def get_litellm_deployment(self, dep_id: int) -> dict | None:
        """Return a single deployment row by id, or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT id, generation, port, pid, status, health_checked_at, "
                "created_at, termination_reason FROM litellm_deployments WHERE id = ?",
                (dep_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "generation", "port", "pid", "status",
                 "health_checked_at", "created_at", "termination_reason"),
                row,
            ))
        finally:
            conn.close()

    def get_active_litellm_deployment(self) -> dict | None:
        """Return the currently active deployment (port, pid, generation), or None.

        Finds the active generation, then returns its most recent active deployment.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT ld.id, ld.generation, ld.port, ld.pid, ld.status, "
                "ld.health_checked_at, ld.created_at "
                "FROM litellm_deployments ld "
                "JOIN litellm_generations lg ON lg.generation = ld.generation "
                "WHERE lg.status = 'active' AND ld.status = 'active' "
                "ORDER BY ld.id DESC LIMIT 1",
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "generation", "port", "pid", "status",
                 "health_checked_at", "created_at"),
                row,
            ))
        finally:
            conn.close()

    def get_litellm_deployment_for_generation(self, generation: int) -> dict | None:
        """Return the deployment for a specific generation, or None.

        Looks up the most recent deployment (active or draining) for the
        requested generation.  This is used when reconstructing an existing
        agent binding that was pinned to a specific LiteLLM generation so
        that its requests continue to be routed to that generation's port
        even after a newer generation has been activated.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT ld.id, ld.generation, ld.port, ld.pid, ld.status, "
                "ld.health_checked_at, ld.created_at "
                "FROM litellm_deployments ld "
                "WHERE ld.generation = ? AND ld.status IN ('active', 'draining') "
                "ORDER BY ld.id DESC LIMIT 1",
                (generation,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "generation", "port", "pid", "status",
                 "health_checked_at", "created_at"),
                row,
            ))
        finally:
            conn.close()

    def store_provider_catalog_entries(
        self, *, provider_id: str, response_digest: str, entries: list[object],
        request_id: str | None = None,
    ) -> None:
        conn = self._new_conn()
        try:
            now = _utcnow()
            conn.execute(
                "INSERT INTO provider_catalog_snapshots "
                "(provider_id, response_digest, fetched_at, request_id, models_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    provider_id,
                    response_digest,
                    now,
                    request_id,
                    json.dumps(
                        [getattr(entry, "raw", {}) for entry in entries],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            for entry in entries:
                model_id = str(getattr(entry, "model_id"))
                endpoint_id = str(getattr(entry, "endpoint_id", "") or "")
                raw = getattr(entry, "raw", {})
                conn.execute(
                    "INSERT INTO provider_catalog_entries "
                    "(provider_id, model_id, endpoint_id, availability, raw_json, discovered_at, response_digest)"
                    " VALUES (?, ?, ?, 'public', ?, ?, ?)"
                    " ON CONFLICT(provider_id, model_id, endpoint_id) DO UPDATE SET"
                    " availability='public', raw_json=excluded.raw_json, discovered_at=excluded.discovered_at,"
                    " response_digest=excluded.response_digest",
                    (provider_id, model_id, endpoint_id, json.dumps(raw, sort_keys=True, separators=(",", ":")), now, response_digest),
                )
            conn.commit()
        finally:
            conn.close()

    def get_provider_catalog_snapshots(self, provider_id: str) -> list[dict]:
        conn = self._new_conn()
        try:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM provider_catalog_snapshots WHERE provider_id=?"
                " ORDER BY generation DESC",
                (provider_id,),
            ).fetchall()]
        finally:
            conn.close()

    def get_provider_catalog(self, provider_id: str) -> list[dict]:
        conn = self._new_conn()
        try:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM provider_catalog_entries WHERE provider_id=? ORDER BY model_id, endpoint_id",
                (provider_id,),
            ).fetchall()]
        finally:
            conn.close()
