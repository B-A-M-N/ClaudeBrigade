"""SQLite-backed RouteState -- run / epoch / route / binding lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enhanced_router.base import DEFAULT_DB_PATH

logger = logging.getLogger("claude-enhanced-router")

SCHEMA_VERSION = 2

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))

_TABLES_DDL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    claude_session_id TEXT,
    cwd TEXT,
    created_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    epoch_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    profile_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','closed')),
    created_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_epoch ON epochs(run_id) WHERE closed_at IS NULL;

CREATE TABLE IF NOT EXISTS role_routes (
    run_id TEXT NOT NULL,
    epoch_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('recon','implementer','adversary','repairer')),
    model_id TEXT NOT NULL,
    source TEXT NOT NULL,
    reason TEXT,
    version INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, epoch_id, role)
);

CREATE TABLE IF NOT EXISTS agent_bindings (
    binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    claude_agent_id TEXT NOT NULL,
    epoch_id TEXT NOT NULL,
    role TEXT NOT NULL,
    model_id TEXT NOT NULL,
    route_version INTEGER NOT NULL,
    bound_at TEXT NOT NULL,
    released_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_agent_binding
    ON agent_bindings(run_id, claude_agent_id) WHERE released_at IS NULL;

CREATE TABLE IF NOT EXISTS route_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    epoch_id TEXT,
    event_type TEXT NOT NULL,
    role TEXT,
    old_model_id TEXT,
    new_model_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_health (
    model_id TEXT NOT NULL,
    configuration_hash TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    status TEXT NOT NULL,
    reachable INTEGER NOT NULL,
    authenticated INTEGER NOT NULL,
    compatible INTEGER NOT NULL,
    failure_rate REAL,
    latency_ms REAL,
    checked_at TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY (model_id, configuration_hash, harness_version)
);
"""

_INDEXES_DDL = """\
CREATE INDEX IF NOT EXISTS idx_role_routes_run_epoch ON role_routes(run_id, epoch_id);
CREATE INDEX IF NOT EXISTS idx_bindings_run_released ON agent_bindings(run_id, claude_agent_id, released_at);
CREATE INDEX IF NOT EXISTS idx_events_run_epoch ON route_events(run_id, epoch_id, id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ Migration --

def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]  # type: ignore[union-attr]
    if current < 1:
        _migrate_v1(conn)
        conn.execute("PRAGMA user_version = 1")
    if current < 2:
        _migrate_v2(conn)
        conn.execute("PRAGMA user_version = 2")


def _migrate_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_TABLES_DDL)
    conn.executescript(_INDEXES_DDL)


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Add binding pinning columns for immutable deployment targets."""
    conn.executescript("""
        ALTER TABLE agent_bindings ADD COLUMN backend TEXT;
        ALTER TABLE agent_bindings ADD COLUMN registry_hash TEXT;
        ALTER TABLE agent_bindings ADD COLUMN catalog_generation INTEGER;
        ALTER TABLE agent_bindings ADD COLUMN litellm_model_name TEXT;
        ALTER TABLE agent_bindings ADD COLUMN upstream_model TEXT;
        ALTER TABLE agent_bindings ADD COLUMN api_base TEXT;
    """)


# ------------------------------------------------------------------ RouteState

class RouteState:
    """SQLite-backed persistent state for runs, epochs, routes, bindings, health."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_migrations(self._new_conn())

    # ---- helpers ---------------------------------------------------

    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    # ---- Run lifecycle ---------------------------------------------

    def create_run(self, run_id: str, session_id: str | None = None, cwd: str | None = None) -> dict:
        """Insert run if not exists (idempotent). Returns run dict."""
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, claude_session_id, cwd, created_at) VALUES (?, ?, ?, ?)",
                (run_id, session_id, cwd, _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT run_id, claude_session_id, cwd, created_at, closed_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to create run {run_id}")
            return dict(zip(
                ("run_id", "claude_session_id", "cwd", "created_at", "closed_at"), row
            ))
        finally:
            conn.close()

    def get_run(self, run_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, claude_session_id, cwd, created_at, closed_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "claude_session_id", "cwd", "created_at", "closed_at"), row
            ))
        finally:
            conn.close()

    def close_run(self, run_id: str) -> None:
        """Set closed_at. Idempotent."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE runs SET closed_at = ? WHERE run_id = ?",
                (_utcnow(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ---- Epoch lifecycle -------------------------------------------

    def get_active_epoch(self, run_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT id, run_id, epoch_id, workflow_id, profile_id, status, created_at, closed_at "
                "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "run_id", "epoch_id", "workflow_id", "profile_id", "status", "created_at", "closed_at"), row
            ))
        finally:
            conn.close()

    def create_epoch(self, run_id: str, epoch_id: str, workflow_id: str, profile_id: str | None = None) -> dict:
        """Create epoch row. Raises ValueError if active epoch already exists."""
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                (run_id,),
            ).fetchone()
            if active:
                raise ValueError(f"Active epoch already exists for run {run_id}")

            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                (run_id, epoch_id, workflow_id, profile_id, _utcnow()),
            )
            conn.commit()
            return self.get_active_epoch(run_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def close_epoch(self, run_id: str, epoch_id: str) -> None:
        """Set closed_at. Release orphaned bindings (released_at IS NULL)."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE epochs SET closed_at = ? WHERE run_id = ? AND epoch_id = ?",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE agent_bindings SET released_at = ? WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                (_utcnow(), run_id, epoch_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_profile_routes_atomic(self, run_id: str, epoch_id: str, profile_id: str, reason: str) -> dict[str, dict]:
        """BEGIN IMMEDIATE transaction: set all 4 role routes from profile. One failure rolls back ALL."""
        # Import registry lazily to avoid circular imports
        from enhanced_router.registry import ModelRegistry

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                now = _utcnow()
                reg = ModelRegistry()
                reg.load_profiles()
                profile = reg.get_profile(profile_id)
                results: dict[str, dict] = {}
                for role in ("recon", "implementer", "adversary", "repairer"):
                    model_id = getattr(profile, role)
                    conn.execute(
                        """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                           VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                           ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                               model_id = excluded.model_id,
                               source = excluded.source,
                               reason = excluded.reason,
                               version = version + 1,
                               changed_at = excluded.changed_at""",
                        (run_id, epoch_id, role, model_id, f"profile:{profile_id}", now),
                    )
                    results[role] = {
                        "run_id": run_id,
                        "epoch_id": epoch_id,
                        "role": role,
                        "model_id": model_id,
                        "source": "profile",
                        "reason": f"profile:{profile_id}",
                        "version": 1,
                    }
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return results
        finally:
            conn.close()

    def create_epoch_from_profile(self, run_id: str, epoch_id: str, workflow_id: str, profile_id: str) -> dict:
        """Transaction: create_epoch + set profile routes. Returns epoch dict."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Check for active epoch
                active = conn.execute(
                    "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                    (run_id,),
                ).fetchone()
                if active:
                    conn.rollback()
                    raise ValueError(f"Active epoch already exists for run {run_id}")

                # Create the epoch row
                conn.execute(
                    "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                    (run_id, epoch_id, workflow_id, profile_id, _utcnow()),
                )

                # Load profile and set routes
                try:
                    from enhanced_router.registry import ModelRegistry
                    reg = ModelRegistry()
                    reg.load_profiles()
                    profile = reg.get_profile(profile_id)
                    now = _utcnow()
                    for role in ("recon", "implementer", "adversary", "repairer"):
                        model_id = getattr(profile, role)
                        conn.execute(
                            """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                               VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                               ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                                   model_id = excluded.model_id,
                                   source = excluded.source,
                                   reason = excluded.reason,
                                   version = version + 1,
                                   changed_at = excluded.changed_at""",
                            (run_id, epoch_id, role, model_id, f"profile:{profile_id}", now),
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to load profile %s for epoch, creating without routes: %s",
                        profile_id, exc,
                    )
                    # Still commit the epoch, routes are optional if profile fails

                conn.commit()
                epoch = conn.execute(
                    "SELECT id, run_id, epoch_id, workflow_id, profile_id, status, created_at, closed_at "
                    "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                    (run_id,),
                ).fetchone()
                if epoch is None:
                    raise RuntimeError(f"Failed to create epoch {epoch_id} for run {run_id}")
                return dict(zip(
                    ("id", "run_id", "epoch_id", "workflow_id", "profile_id", "status", "created_at", "closed_at"),
                    epoch,
                ))
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    # ---- Route operations ------------------------------------------

    def set_role_route(self, run_id: str, epoch_id: str, role: str, model_id: str, source: str, reason: str = "") -> dict:
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
                """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                       model_id = excluded.model_id,
                       source = excluded.source,
                       reason = excluded.reason,
                       version = excluded.version,
                       changed_at = excluded.changed_at""",
                (run_id, epoch_id, role, model_id, source, reason, new_version, now),
            )

            conn.execute(
                "INSERT INTO route_events (run_id, epoch_id, event_type, role, old_model_id, new_model_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, "route_change", role, old_model_id, model_id, now),
            )
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
            }
        finally:
            conn.close()

    def get_role_route(self, run_id: str, epoch_id: str, role: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at"), row
            ))
        finally:
            conn.close()

    def get_epoch_routes(self, run_id: str, epoch_id: str) -> dict[str, dict]:
        """Return dict of {role: {model_id, version, ...}}."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? ORDER BY role",
                (run_id, epoch_id),
            ).fetchall()
            result: dict[str, dict] = {}
            for row in rows:
                role = row[2]
                result[role] = dict(zip(
                    ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at"), row
                ))
            return result
        finally:
            conn.close()

    # ---- Binding operations ----------------------------------------

    def bind_agent(
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
    ) -> int:
        """Create binding row. Raises ValueError if active binding exists. Returns binding_id."""
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT binding_id FROM agent_bindings WHERE run_id = ? AND claude_agent_id = ? AND released_at IS NULL",
                (run_id, claude_agent_id),
            ).fetchone()
            if active:
                raise ValueError(
                    f"Active binding already exists for agent {claude_agent_id} in run {run_id}"
                )

            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            cursor = conn.execute(
                "INSERT INTO agent_bindings "
                "(run_id, claude_agent_id, epoch_id, role, model_id, route_version, bound_at, "
                " backend, registry_hash, catalog_generation, litellm_model_name, upstream_model, api_base) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, claude_agent_id, epoch_id, role, model_id, route_version, now,
                    backend, registry_hash, catalog_generation, litellm_model_name, upstream_model, api_base,
                ),
            )
            binding_id: int = cursor.lastrowid  # type: ignore[assignment]
            conn.commit()
            return binding_id
        finally:
            conn.close()

    def get_agent_binding(self, run_id: str, claude_agent_id: str) -> dict | None:
        """Return active binding (released_at IS NULL) or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT binding_id, run_id, claude_agent_id, epoch_id, role, model_id, "
                "route_version, bound_at, released_at, backend, registry_hash, "
                "catalog_generation, litellm_model_name, upstream_model, api_base "
                "FROM agent_bindings WHERE run_id = ? AND claude_agent_id = ? AND released_at IS NULL",
                (run_id, claude_agent_id),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("binding_id", "run_id", "claude_agent_id", "epoch_id", "role", "model_id",
                 "route_version", "bound_at", "released_at", "backend", "registry_hash",
                 "catalog_generation", "litellm_model_name", "upstream_model", "api_base"),
                row,
            ))
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
                    "SELECT binding_id, run_id, claude_agent_id, epoch_id, role, model_id, "
                    "route_version, bound_at, released_at, backend, registry_hash, "
                    "catalog_generation, litellm_model_name, upstream_model, api_base "
                    "FROM agent_bindings WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                    (run_id, epoch_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT binding_id, run_id, claude_agent_id, epoch_id, role, model_id, "
                    "route_version, bound_at, released_at, backend, registry_hash, "
                    "catalog_generation, litellm_model_name, upstream_model, api_base "
                    "FROM agent_bindings WHERE run_id = ? AND released_at IS NULL",
                    (run_id,),
                ).fetchall()
            return [
                dict(zip(
                    ("binding_id", "run_id", "claude_agent_id", "epoch_id", "role", "model_id",
                     "route_version", "bound_at", "released_at", "backend", "registry_hash",
                     "catalog_generation", "litellm_model_name", "upstream_model", "api_base"),
                    row,
                ))
                for row in rows
            ]
        finally:
            conn.close()

    # ---- Snapshot --------------------------------------------------

    def create_route_snapshot(self, run_id: str, epoch_id: str, purpose: str = "completion") -> str:
        """Capture state as JSON, SHA-256. Returns hex digest."""
        conn = self._new_conn()
        try:
            routes_rows = conn.execute(
                "SELECT role, model_id, version FROM role_routes WHERE run_id = ? AND epoch_id = ?",
                (run_id, epoch_id),
            ).fetchall()
            routes = {row[0]: {"model_id": row[1], "version": row[2]} for row in routes_rows}

            bindings_rows = conn.execute(
                "SELECT claude_agent_id, role, model_id, binding_id FROM agent_bindings "
                "WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                (run_id, epoch_id),
            ).fetchall()
            bindings = {
                row[0]: {"role": row[1], "model_id": row[2], "binding_id": row[3]}
                for row in bindings_rows
            }

            max_event = conn.execute(
                "SELECT MAX(id) FROM route_events WHERE run_id = ? AND epoch_id = ?",
                (run_id, epoch_id),
            ).fetchone()[0]  # type: ignore[index]

            snapshot_data = {
                "purpose": purpose,
                "run_id": run_id,
                "epoch_id": epoch_id,
                "routes": routes,
                "bindings": bindings,
                "max_event_id": max_event,
                "captured_at": _utcnow(),
            }

            json_bytes = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(json_bytes).hexdigest()
        finally:
            conn.close()

    # ---- Health ----------------------------------------------------

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


# ------------------------------------------------------------------ Singleton

_state: RouteState | None = None


def get_state() -> RouteState:
    global _state
    if _state is None:
        _state = RouteState(DEFAULT_DB_PATH)
    return _state
