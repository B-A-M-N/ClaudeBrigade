"""Agent binding persistence, split out of state.py.

Sixth increment of the incremental extraction out of ``RouteState``: these
methods only touch the ``agent_bindings`` table (plus a read-only join
against ``litellm_deployments``) through ``self._new_conn()``, so they move
to their own module as a mixin without touching any call site.
``RouteState`` still exposes these methods under their original names.

``_select_agent_binding`` is also called directly from other sections still
in ``state.py`` (run lifecycle, agent execution lifecycle) -- that keeps
working unchanged since it resolves through ``self`` via the mixin's normal
method resolution order, exactly like the public methods below.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class BindingRepository:
    """Mixin providing agent binding persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    @staticmethod
    def _select_agent_binding(
        conn: sqlite3.Connection,
        run_id: str,
        claude_agent_id: str,
    ) -> dict | None:
        row = conn.execute(
            "SELECT ab.*, "
            "(SELECT ld.port FROM litellm_deployments AS ld "
            " WHERE ld.generation = ab.catalog_generation "
            " AND ld.status IN ('active', 'draining') "
            " ORDER BY ld.id DESC LIMIT 1) AS litellm_port, "
            "(SELECT ld.status FROM litellm_deployments AS ld "
            " WHERE ld.generation = ab.catalog_generation "
            " AND ld.status IN ('active', 'draining') "
            " ORDER BY ld.id DESC LIMIT 1) AS litellm_deployment_status "
            "FROM agent_bindings AS ab "
            "WHERE ab.run_id=? AND ab.claude_agent_id=? AND ab.released_at IS NULL",
            (run_id, claude_agent_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def bind_or_get_agent(
        self,
        run_id: str,
        claude_agent_id: str,
        epoch_id: str,
        role: str,
        model_id: str,
        route_version: int,
        backend: str | None = None,
        registry_hash: str | None = None,
        catalog_generation: int | None = None,
        litellm_model_name: str | None = None,
        upstream_model: str | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        auth_spec_json: str | None = None,
        endpoint_id: str | None = None,
        endpoint_selection_reason: str | None = None,
        endpoint_policy_json: str | None = None,
        configuration_hash: str | None = None,
        certification_id: str | None = None,
        provider_id: str | None = None,
        provider_ids_json: str | None = None,
        routing_mode: str = "fixed",
        deployment_group: str | None = None,
        allowed_deployments_json: str | None = None,
        deployment_policy_digest: str | None = None,
        claude_session_id: str | None = None,
        claude_parent_agent_id: str | None = None,
    ) -> tuple[dict, bool]:
        """Atomically bind or return existing binding.

        Runs inside one ``BEGIN IMMEDIATE`` transaction:

        1. Check for existing active binding
        2. If found, return ``(binding_dict, False)``
        3. If not found, INSERT and return ``(new_binding_dict, True)``

        This eliminates the race where two simultaneous first-request
        resolutions both observe no binding before the INSERT.

        Returns
        -------
        (binding_dict, is_new)
            *binding_dict* is a dict with binding details.
            *is_new* is ``True`` when a new binding was created.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._select_agent_binding(conn, run_id, claude_agent_id)
                if existing is not None:
                    conn.commit()
                    return (existing, False)

                now = _utcnow()
                conn.execute(
                    "INSERT INTO agent_bindings "
                    "(run_id, claude_agent_id, epoch_id, role, model_id, route_version, bound_at, "
                    " backend, registry_hash, catalog_generation, litellm_model_name, upstream_model, api_base, api_key_env,"
                    " auth_spec_json, endpoint_id, endpoint_selection_reason, endpoint_policy_json, certification_id, provider_id,"
                    " provider_ids_json, claude_session_id, claude_parent_agent_id, configuration_hash, routing_mode, deployment_group,"
                    " allowed_deployments_json, deployment_policy_digest) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, claude_agent_id, epoch_id, role, model_id, route_version, now,
                        backend, registry_hash, catalog_generation, litellm_model_name, upstream_model, api_base, api_key_env,
                        auth_spec_json or "", endpoint_id, endpoint_selection_reason, endpoint_policy_json,
                        certification_id, provider_id, provider_ids_json, claude_session_id or "", claude_parent_agent_id or "",
                        configuration_hash, routing_mode, deployment_group, allowed_deployments_json,
                        deployment_policy_digest,
                    ),
                )
                binding = self._select_agent_binding(conn, run_id, claude_agent_id)
                assert binding is not None  # just inserted
                conn.commit()
                return (binding, True)
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def get_agent_binding(self, run_id: str, claude_agent_id: str) -> dict | None:
        """Return active binding (released_at IS NULL) or None."""
        conn = self._new_conn()
        try:
            return self._select_agent_binding(conn, run_id, claude_agent_id)
        finally:
            conn.close()

    def release_binding(self, run_id: str, claude_agent_id: str) -> None:
        """Set released_at on active binding. Idempotent. Does NOT delete history."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE agent_bindings SET released_at = ? WHERE run_id = ? AND claude_agent_id = ? AND released_at IS NULL",
                (_utcnow(), run_id, claude_agent_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active_bindings(self, run_id: str, epoch_id: str | None = None) -> list[dict]:
        conn = self._new_conn()
        try:
            if epoch_id:
                rows = conn.execute(
                    "SELECT ab.* "
                    "FROM agent_bindings AS ab WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                    (run_id, epoch_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ab.* "
                    "FROM agent_bindings AS ab WHERE run_id = ? AND released_at IS NULL",
                    (run_id,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
