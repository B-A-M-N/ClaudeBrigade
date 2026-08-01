"""Role route persistence, split out of state.py.

Fifth increment of the incremental extraction out of ``RouteState``: these
methods only touch the ``role_routes``/``route_events`` tables through
``self._new_conn()``, so they move to their own module as a mixin without
touching any call site. ``RouteState`` still exposes these methods under
their original names.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RouteOperationsRepository:
    """Mixin providing role-route persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def set_role_route(
        self, run_id: str, epoch_id: str, role: str, model_id: str, source: str,
        reason: str = "", endpoint_override: str | None = None,
        fallback_models: list[str] | None = None,
        fallback_routes: list[dict] | None = None,
    ) -> dict:
        """Upsert role_routes row (increment version). Append route_events. Returns route dict."""
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {_VALID_ROLES}")

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()

            # Fetch current row so we know old model_id and version
            existing = conn.execute(
                "SELECT version, model_id FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()

            old_model_id = existing[1] if existing else None
            new_version = (existing[0] if existing else 0) + 1

            conn.execute(
                """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at, endpoint_id, endpoint_override)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                       model_id = excluded.model_id,
                       source = excluded.source,
                       reason = excluded.reason,
                       version = excluded.version,
                       changed_at = excluded.changed_at,
                       endpoint_override = excluded.endpoint_override""",
                (run_id, epoch_id, role, model_id, source, reason, new_version, now, endpoint_override, endpoint_override),
            )

            conn.execute(
                "INSERT INTO route_events (run_id, epoch_id, event_type, role, old_model_id, new_model_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, "route_change", role, old_model_id, model_id, now),
            )
            if fallback_routes is not None:
                derived_fallback_models = [
                    c["model"] for c in fallback_routes if isinstance(c, dict) and c.get("model")
                ]
                conn.execute(
                    "UPDATE role_routes SET fallback_models_json=?, fallback_routes_json=? "
                    "WHERE run_id=? AND epoch_id=? AND role=?",
                    (json.dumps(derived_fallback_models), json.dumps(fallback_routes), run_id, epoch_id, role),
                )
                fallback_models = derived_fallback_models
            elif fallback_models is not None:
                derived_fallback_routes = [{"model": m, "endpoint": "auto"} for m in fallback_models]
                conn.execute(
                    "UPDATE role_routes SET fallback_models_json=?, fallback_routes_json=? "
                    "WHERE run_id=? AND epoch_id=? AND role=?",
                    (json.dumps(fallback_models), json.dumps(derived_fallback_routes), run_id, epoch_id, role),
                )
                fallback_routes = derived_fallback_routes
            conn.commit()

            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "role": role,
                "model_id": model_id,
                "source": source,
                "reason": reason,
                "version": new_version,
                "changed_at": now,
                "endpoint_override": endpoint_override,
                "fallback_models": fallback_models or [],
                "fallback_routes": fallback_routes or [],
            }
        finally:
            conn.close()

    _ROLE_ROUTE_COLUMNS: tuple[str, ...] = (
        "run_id", "epoch_id", "role", "model_id", "source", "reason", "version",
        "changed_at", "endpoint_id", "endpoint_override", "fallback_models_json",
        "fallback_routes_json",
    )

    @classmethod
    def _row_to_role_route(cls, row) -> dict:
        route: dict = dict(zip(cls._ROLE_ROUTE_COLUMNS, row))
        try:
            route["fallback_routes"] = json.loads(route.get("fallback_routes_json") or "[]")
        except (TypeError, ValueError):
            route["fallback_routes"] = []
        if not isinstance(route["fallback_routes"], list):
            route["fallback_routes"] = []
        return route

    def get_role_route(self, run_id: str, epoch_id: str, role: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                f"SELECT {', '.join(self._ROLE_ROUTE_COLUMNS)} "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_role_route(row)
        finally:
            conn.close()

    def get_epoch_routes(self, run_id: str, epoch_id: str) -> dict[str, dict]:
        """Return dict of {role: {model_id, version, ...}}."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(self._ROLE_ROUTE_COLUMNS)} "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? ORDER BY role",
                (run_id, epoch_id),
            ).fetchall()
            result: dict[str, dict] = {}
            for row in rows:
                route = self._row_to_role_route(row)
                result[route["role"]] = route
            return result
        finally:
            conn.close()
