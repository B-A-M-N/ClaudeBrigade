"""Controller binding and endpoint observation persistence, split out of
state.py.

Eighth increment of the incremental extraction out of ``RouteState``: these
methods only touch ``controller_bindings`` (plus a read-only join against
``litellm_deployments``), ``model_endpoint_usage``, and
``endpoint_certifications`` through ``self._new_conn()``, so they move to
their own module as a mixin without touching any call site. ``RouteState``
still exposes these methods under their original names; other sections that
call ``self.get_controller_binding(...)`` keep working unchanged via the
mixin's normal method resolution order.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import sqlite3
from datetime import datetime, timedelta, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_age(max_age_seconds: int) -> str:
    """Return an ISO-8601 timestamp that is *max_age_seconds* in the past."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    return cutoff.isoformat()


class ControllerBindingRepository(RepositoryMixin):
    """Mixin providing controller binding/endpoint observation methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    @staticmethod
    def _select_controller_binding(
        conn: sqlite3.Connection,
        run_id: str,
        client_session_id: str,
    ) -> dict | None:
        row = conn.execute(
            "SELECT cb.*, "
            "(SELECT ld.port FROM litellm_deployments AS ld "
            " WHERE ld.generation = cb.catalog_generation "
            " AND ld.status IN ('active', 'draining') "
            " ORDER BY ld.id DESC LIMIT 1) AS litellm_port, "
            "(SELECT ld.status FROM litellm_deployments AS ld "
            " WHERE ld.generation = cb.catalog_generation "
            " AND ld.status IN ('active', 'draining') "
            " ORDER BY ld.id DESC LIMIT 1) AS litellm_deployment_status "
            "FROM controller_bindings AS cb "
            "WHERE cb.run_id=? AND cb.client_session_id=? AND cb.released_at IS NULL",
            (run_id, client_session_id),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_controller_binding(self, run_id: str, client_session_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            return self._select_controller_binding(conn, run_id, client_session_id)
        finally:
            conn.close()

    def bind_or_get_controller(
        self,
        *,
        run_id: str,
        client_session_id: str,
        public_model: str,
        registry_model_id: str,
        backend: str,
        upstream_model: str | None,
        provider_id: str | None,
        api_base: str | None,
        catalog_generation: int | None,
        registry_hash: str,
        certification_id: str | None,
        auth_spec_json: str | None,
        api_key_env: str | None,
        provider_ids_json: str | None = None,
        endpoint_id: str | None = None,
        endpoint_selection_reason: str | None = None,
        endpoint_policy_json: str | None = None,
        litellm_model_name: str | None = None,
        configuration_hash: str | None = None,
        routing_mode: str = "fixed",
        deployment_group: str | None = None,
        allowed_deployments_json: str | None = None,
        deployment_policy_digest: str | None = None,
        route_digest: str | None = None,
        candidate_index: int | None = None,
    ) -> tuple[dict, bool]:
        """Create the immutable controller binding for one client session."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._select_controller_binding(conn, run_id, client_session_id)
            if existing is not None:
                conn.commit()
                return existing, False
            now = _utcnow()
            conn.execute(
                "INSERT INTO controller_bindings "
                "(run_id, client_session_id, public_model, registry_model_id, backend, upstream_model,"
                " provider_id, api_base, catalog_generation, registry_hash, certification_id, auth_spec_json,"
                " api_key_env, endpoint_id, provider_ids_json, endpoint_selection_reason, endpoint_policy_json,"
                " litellm_model_name, configuration_hash, routing_mode, deployment_group,"
                " allowed_deployments_json, deployment_policy_digest, route_digest, candidate_index, bound_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, client_session_id, public_model, registry_model_id, backend, upstream_model,
                 provider_id, api_base, catalog_generation, registry_hash, certification_id, auth_spec_json,
                 api_key_env, endpoint_id, provider_ids_json, endpoint_selection_reason, endpoint_policy_json,
                litellm_model_name, configuration_hash, routing_mode, deployment_group,
                allowed_deployments_json, deployment_policy_digest, route_digest, candidate_index, now),
            )
            binding = self._select_controller_binding(conn, run_id, client_session_id)
            assert binding is not None
            conn.commit()
            return binding, True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_endpoint_usage(
        self,
        *,
        provider_id: str,
        model_id: str,
        endpoint_id: str,
        request_id: str | None,
        input_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float | None = None,
        succeeded: bool = True,
        configuration_hash: str = "",
    ) -> None:
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO model_endpoint_usage "
                "(provider_id, model_id, endpoint_id, request_id, input_tokens_total, cache_read_tokens,"
                " cache_write_tokens, output_tokens, latency_ms, succeeded, configuration_hash, observed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (provider_id, model_id, endpoint_id, request_id, max(0, input_tokens), max(0, cache_read_tokens),
                 max(0, cache_write_tokens), max(0, output_tokens), latency_ms, int(succeeded),
                 configuration_hash, _utcnow()),
            )
            conn.commit()
        finally:
            conn.close()

    def is_endpoint_certified(
        self,
        *,
        provider_id: str,
        model_id: str,
        endpoint_id: str,
        configuration_hash: str,
        capabilities: tuple[str, ...] = ("messages", "streaming"),
    ) -> bool:
        """Return true only when every required capability has fresh evidence."""
        conn = self._new_conn()
        try:
            for capability in capabilities:
                row = conn.execute(
                    "SELECT 1 FROM endpoint_certifications "
                    "WHERE provider_id=? AND model_id=? AND endpoint_id=? "
                    "AND configuration_hash=? AND capability=? AND status='pass' "
                    "AND (expires_at IS NULL OR expires_at > ?)",
                    (provider_id, model_id, endpoint_id, configuration_hash, capability, _utcnow()),
                ).fetchone()
                if row is None:
                    return False
            return True
        finally:
            conn.close()

    def record_endpoint_certifications(
        self,
        *,
        provider_id: str,
        model_id: str,
        endpoint_id: str,
        configuration_hash: str,
        harness_version: str,
        protocol_version: str,
        evidence_digest: str,
        capabilities: dict[str, bool],
        certification_id: str | None = None,
        litellm_version: str | None = None,
        expires_at: str | None = None,
    ) -> list[dict]:
        """Publish capability-specific endpoint certification evidence.

        A provider catalog and a successful HTTP response do not themselves
        certify Claude Code tool compatibility.  Callers publish one result
        per capability after the compatibility harness completes.
        """
        if not capabilities:
            raise ValueError("at least one endpoint capability is required")
        prefix = certification_id or f"cert-{provider_id}-{model_id}-{endpoint_id}"
        now = _utcnow()
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for capability, passed in sorted(capabilities.items()):
                cid = f"{prefix}-{capability}"
                conn.execute(
                    "INSERT INTO endpoint_certifications "
                    "(certification_id, provider_id, model_id, endpoint_id, configuration_hash,"
                    " litellm_version, harness_version, protocol_version, capability, status,"
                    " certified_at, expires_at, evidence_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(provider_id, model_id, endpoint_id, configuration_hash, capability)"
                    " DO UPDATE SET certification_id=excluded.certification_id,"
                    " litellm_version=excluded.litellm_version, harness_version=excluded.harness_version,"
                    " protocol_version=excluded.protocol_version, status=excluded.status,"
                    " certified_at=excluded.certified_at, expires_at=excluded.expires_at,"
                    " evidence_digest=excluded.evidence_digest",
                    (
                        cid, provider_id, model_id, endpoint_id, configuration_hash,
                        litellm_version, harness_version, protocol_version, capability,
                        "pass" if passed else "fail", now, expires_at, evidence_digest,
                    ),
                )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM endpoint_certifications WHERE provider_id=? AND model_id=?"
                " AND endpoint_id=? AND configuration_hash=? ORDER BY capability",
                (provider_id, model_id, endpoint_id, configuration_hash),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_endpoint_certifications(
        self,
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
        endpoint_id: str | None = None,
        configuration_hash: str | None = None,
    ) -> list[dict]:
        """Return certification evidence for health and route inspection."""
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("provider_id", provider_id),
            ("model_id", model_id),
            ("endpoint_id", endpoint_id),
            ("configuration_hash", configuration_hash),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        query = "SELECT * FROM endpoint_certifications"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY provider_id, model_id, endpoint_id, capability"
        conn = self._new_conn()
        try:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def get_endpoint_observations(
        self,
        model_id: str | None = None,
        *,
        provider_id: str | None = None,
        configuration_hash: str | None = None,
        maximum_age_seconds: int | None = None,
    ) -> dict[str, dict]:
        """Return endpoint aggregates for one immutable configuration identity."""
        conn = self._new_conn()
        try:
            clauses = ["model_id=?"]
            params: list[object] = [model_id]
            if provider_id is not None:
                clauses.append("provider_id=?")
                params.append(provider_id)
            if configuration_hash is not None:
                clauses.append("configuration_hash=?")
                params.append(configuration_hash)
            if maximum_age_seconds is not None:
                clauses.append("observed_at >= ?")
                params.append(_utcnow_age(maximum_age_seconds))
            rows = conn.execute(
                "SELECT endpoint_id, COUNT(*) AS sample_count, SUM(input_tokens_total) AS input_tokens_total,"
                " SUM(cache_read_tokens) AS cache_read_tokens, SUM(succeeded) AS succeeded,"
                " MAX(observed_at) AS observed_at FROM model_endpoint_usage"
                f" WHERE {' AND '.join(clauses)} GROUP BY endpoint_id",
                params,
            ).fetchall()
            result: dict[str, dict] = {}
            for row in rows:
                item = dict(row)
                total = int(item.get("input_tokens_total") or 0)
                item["cache_rate"] = (int(item.get("cache_read_tokens") or 0) / total) if total else 0.0
                item["success_rate"] = (int(item.get("succeeded") or 0) / int(item["sample_count"])) if item.get("sample_count") else 0.0
                result[item["endpoint_id"]] = item
            return result
        finally:
            conn.close()
