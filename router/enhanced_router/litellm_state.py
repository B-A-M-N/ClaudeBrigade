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

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiteLLMGenerationRepository(RepositoryMixin):
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

    def record_litellm_deployment_event(
        self,
        *,
        deployment_id: int,
        generation: int,
        event: str,
        status: str | None = None,
        pid: int | None = None,
        port: int | None = None,
        active_requests: int = 0,
        active_streams: int = 0,
        reason: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Persist one bounded LiteLLM deployment lifecycle observation."""
        if not event.strip():
            raise ValueError("deployment telemetry event must not be empty")
        conn = self._new_conn()
        try:
            now = _utcnow()
            cursor = conn.execute(
                "INSERT INTO litellm_deployment_events "
                "(deployment_id, generation, event, status, pid, port, "
                "active_requests, active_streams, reason, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    deployment_id,
                    generation,
                    event,
                    status,
                    pid,
                    port,
                    max(0, int(active_requests)),
                    max(0, int(active_streams)),
                    (reason or "")[:500] or None,
                    json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")),
                    now,
                ),
            )
            conn.commit()
            event_id = int(cursor.lastrowid or 0)
            row = conn.execute(
                "SELECT * FROM litellm_deployment_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            return dict(row) if row is not None else {"event_id": event_id}
        finally:
            conn.close()

    def get_litellm_deployment_events(
        self, deployment_id: int, *, limit: int = 100,
    ) -> list[dict]:
        """Return recent lifecycle observations for one deployment."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM litellm_deployment_events WHERE deployment_id=? "
                "ORDER BY event_id DESC LIMIT ?",
                (deployment_id, max(1, min(int(limit), 1000))),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_litellm_deployment_telemetry(
        self, *, deployment_id: int | None = None, limit: int = 100,
    ) -> list[dict]:
        """Return deployment rows enriched with last activity and event count."""
        conn = self._new_conn()
        try:
            where = "WHERE ld.id=?" if deployment_id is not None else ""
            params: tuple[object, ...] = (deployment_id,) if deployment_id is not None else ()
            rows = conn.execute(
                "SELECT ld.*, lg.status AS generation_status, "
                "(SELECT COUNT(*) FROM litellm_deployment_events e "
                " WHERE e.deployment_id=ld.id) AS event_count, "
                "(SELECT e.event FROM litellm_deployment_events e "
                " WHERE e.deployment_id=ld.id ORDER BY e.event_id DESC LIMIT 1) AS last_event, "
                "(SELECT e.active_requests FROM litellm_deployment_events e "
                " WHERE e.deployment_id=ld.id ORDER BY e.event_id DESC LIMIT 1) AS last_active_requests, "
                "(SELECT e.active_streams FROM litellm_deployment_events e "
                " WHERE e.deployment_id=ld.id ORDER BY e.event_id DESC LIMIT 1) AS last_active_streams "
                "FROM litellm_deployments ld "
                "JOIN litellm_generations lg ON lg.generation=ld.generation "
                f"{where} ORDER BY ld.id DESC LIMIT ?",
                (*params, max(1, min(int(limit), 1000))),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def record_litellm_request_attribution(
        self,
        *,
        request_id: str,
        generation: int | None,
        agent_binding_id: int | None,
        logical_model_id: str | None,
        requested_provider_ids: list[str] | tuple[str, ...],
        allowed_deployments: list[str] | tuple[str, ...],
        reported_deployment_id: str | None,
        actual_provider_id: str | None,
        actual_endpoint_id: str | None,
        attribution_source: str,
        trusted: bool,
        status_code: int | None,
    ) -> dict:
        """Record one response's physical-deployment attribution.

        ``trusted`` is set by the router only after the response identity is
        present in the binding snapshot's allowed deployment set.  This table
        intentionally retains both the requested candidate set and the
        observed identity so operators can audit conservative admission.
        """
        if not request_id.strip():
            raise ValueError("LiteLLM attribution requires a request ID")
        if not attribution_source.strip():
            raise ValueError("LiteLLM attribution requires a source")
        conn = self._new_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO litellm_request_attributions ("
                "request_id, generation, agent_binding_id, logical_model_id, "
                "requested_provider_ids_json, allowed_deployments_json, "
                "reported_deployment_id, actual_provider_id, actual_endpoint_id, "
                "attribution_source, trusted, status_code, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    generation,
                    agent_binding_id,
                    logical_model_id,
                    json.dumps(sorted(set(requested_provider_ids)), separators=(",", ":")),
                    json.dumps(sorted(set(allowed_deployments)), separators=(",", ":")),
                    reported_deployment_id,
                    actual_provider_id,
                    actual_endpoint_id,
                    attribution_source,
                    int(trusted),
                    status_code,
                    _utcnow(),
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM litellm_request_attributions WHERE attribution_id=?",
                (int(cursor.lastrowid or 0),),
            ).fetchone()
            return dict(row) if row is not None else {}
        finally:
            conn.close()

    def get_litellm_request_attributions(
        self, *, request_id: str | None = None, limit: int = 100,
    ) -> list[dict]:
        """Return recent physical-deployment attribution observations."""
        conn = self._new_conn()
        try:
            if request_id:
                rows = conn.execute(
                    "SELECT * FROM litellm_request_attributions "
                    "WHERE request_id=? ORDER BY attribution_id DESC LIMIT ?",
                    (request_id, max(1, min(int(limit), 1000))),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM litellm_request_attributions "
                    "ORDER BY attribution_id DESC LIMIT ?",
                    (max(1, min(int(limit), 1000)),),
                ).fetchall()
            return [dict(row) for row in rows]
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
