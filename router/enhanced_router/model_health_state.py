"""Model health/certification persistence, split out of state.py.

Second increment of the incremental extraction out of ``RouteState``: these
four methods only touch the ``model_health``/``model_certification`` tables
through ``self._new_conn()``, so -- like the LiteLLM generation repository
before it -- they move to their own module as a mixin without touching any
call site. ``RouteState`` still exposes these methods under their original
names.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelHealthRepository:
    """Mixin providing model health/certification persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def set_model_health(
        self,
        model_id: str,
        config_hash: str,
        harness_version: str,
        status: str,
        reachable: bool,
        authenticated: bool,
        compatible: bool,
        failure_rate: float | None = None,
        latency_ms: float | None = None,
        reason: str = "",
    ) -> None:
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO model_health
                   (model_id, configuration_hash, harness_version, status, reachable, authenticated, compatible,
                    failure_rate, latency_ms, checked_at, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(model_id, configuration_hash, harness_version) DO UPDATE SET
                       status = excluded.status,
                       reachable = excluded.reachable,
                       authenticated = excluded.authenticated,
                       compatible = excluded.compatible,
                       failure_rate = excluded.failure_rate,
                       latency_ms = excluded.latency_ms,
                       checked_at = excluded.checked_at,
                       reason = excluded.reason""",
                (
                    model_id, config_hash, harness_version, status,
                    int(reachable), int(authenticated), int(compatible),
                    failure_rate, latency_ms, _utcnow(), reason,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_model_health(self, model_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT model_id, configuration_hash, harness_version, status, reachable, authenticated, "
                "compatible, failure_rate, latency_ms, checked_at, reason "
                "FROM model_health WHERE model_id = ? ORDER BY checked_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("model_id", "configuration_hash", "harness_version", "status", "reachable", "authenticated",
                 "compatible", "failure_rate", "latency_ms", "checked_at", "reason"),
                row,
            ))
        finally:
            conn.close()

    def set_model_certification(
        self,
        certification_id: str,
        model_id: str,
        configuration_hash: str,
        harness_version: str,
        protocol_version: str,
        tool_call_pass: bool,
        streaming_pass: bool,
        cancellation_pass: bool,
        parallel_tool_behavior: str | None = None,
        max_validated_context: int | None = None,
        provider_endpoint_digest: str | None = None,
        expires_at: str | None = None,
        notes: str = "",
    ) -> None:
        """Record a model certification result from the compatibility harness."""
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO model_certification
                   (certification_id, model_id, configuration_hash, harness_version, protocol_version,
                    tool_call_pass, streaming_pass, cancellation_pass, parallel_tool_behavior,
                    max_validated_context, provider_endpoint_digest, expires_at, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(certification_id) DO UPDATE SET
                       model_id = excluded.model_id,
                       configuration_hash = excluded.configuration_hash,
                       harness_version = excluded.harness_version,
                       protocol_version = excluded.protocol_version,
                       tool_call_pass = excluded.tool_call_pass,
                       streaming_pass = excluded.streaming_pass,
                       cancellation_pass = excluded.cancellation_pass,
                       parallel_tool_behavior = excluded.parallel_tool_behavior,
                       max_validated_context = excluded.max_validated_context,
                       provider_endpoint_digest = excluded.provider_endpoint_digest,
                       expires_at = excluded.expires_at,
                       notes = excluded.notes""",
                (
                    certification_id, model_id, configuration_hash, harness_version, protocol_version,
                    int(tool_call_pass), int(streaming_pass), int(cancellation_pass), parallel_tool_behavior,
                    max_validated_context, provider_endpoint_digest, expires_at, notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_model_certification(self, model_id: str) -> dict | None:
        """Get the latest certification for a model."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM model_certification WHERE model_id = ? ORDER BY certified_at DESC LIMIT 1",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            cols = [c[0] for c in conn.execute("PRAGMA table_info(model_certification)")]
            return dict(zip(cols, row))
        finally:
            conn.close()
