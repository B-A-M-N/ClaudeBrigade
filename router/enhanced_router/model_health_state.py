"""Model health/certification persistence, split out of state.py.

Second increment of the incremental extraction out of ``RouteState``: these
four methods only touch the ``model_health``/``model_certification`` tables
through ``self._new_conn()``, so -- like the LiteLLM generation repository
before it -- they move to their own module as a mixin without touching any
call site. ``RouteState`` still exposes these methods under their original
names.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import sqlite3
from datetime import datetime, timezone


def evaluate_model_health(
    row: sqlite3.Row | dict | None,
    *,
    max_age_seconds: float = 900.0,
    allow_untested: bool = True,
    now: datetime | None = None,
) -> dict[str, object]:
    """Evaluate whether a persisted health result is safe for admission.

    Health records are intentionally conservative at the admission boundary:
    an absent record is ``untested`` (configurable), a malformed or old record
    is ``stale``, and a record whose probe failed is ``unhealthy``.  The
    function is pure so reservation promotion can evaluate rows inside its
    existing SQLite transaction without opening a second connection.
    """
    if row is None:
        return {
            "admissible": bool(allow_untested),
            "status": "untested",
            "reason": "no model health check has been recorded",
            "age_seconds": None,
        }

    checked_at = row["checked_at"] if isinstance(row, sqlite3.Row) else row.get("checked_at")
    current = now or datetime.now(timezone.utc)
    age_seconds: float | None = None
    if not checked_at:
        status = "stale"
        reason = "model health record has no checked_at timestamp"
    else:
        try:
            stamp = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (current - stamp).total_seconds())
        except ValueError:
            age_seconds = None
        if age_seconds is None or age_seconds > max_age_seconds:
            status = "stale"
            reason = f"model health check is older than {max_age_seconds:g}s"
        else:
            reachable = bool(row["reachable"] if isinstance(row, sqlite3.Row) else row.get("reachable"))
            authenticated = bool(
                row["authenticated"] if isinstance(row, sqlite3.Row) else row.get("authenticated")
            )
            compatible = bool(row["compatible"] if isinstance(row, sqlite3.Row) else row.get("compatible"))
            if not (reachable and authenticated and compatible):
                status = "unhealthy"
                reason = "latest model health check failed reachability, authentication, or compatibility"
            else:
                status = str(row["status"] if isinstance(row, sqlite3.Row) else row.get("status") or "healthy")
                reason = "latest model health check is fresh and passed"

    return {
        "admissible": status not in {"stale", "unhealthy"},
        "status": status,
        "reason": reason,
        "age_seconds": age_seconds,
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelHealthRepository(RepositoryMixin):
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
