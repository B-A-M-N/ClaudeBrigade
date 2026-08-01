"""SQLite-backed RouteState -- run / epoch / route / binding lifecycle."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Callable

from enhanced_router.base import DEFAULT_DB_PATH

logger = logging.getLogger("claude-enhanced-router")

# Keep the public schema version stable for existing Brigade installations;
# the hardening migration below is shape-detected so databases already at v36
# are upgraded without forcing a second version bump.
SCHEMA_VERSION = 36

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))

VALID_COMMAND_TYPES = frozenset((
    "model_change", "profile_set", "route_change",
    "binding_release", "binding_reenable",
))

# The current SQLite schema predates the richer cross-lane execution states.
# Keep its wire-compatible ``timeout`` spelling for now, while accepting the
# audit's ``timed_out`` spelling at the API boundary.  The important invariant
# here is that a terminal execution cannot silently become active again.
_EXECUTION_STATUSES = frozenset((
    "started", "running", "completed", "failed", "timeout", "cancelled",
))
_EXECUTION_TERMINAL_STATUSES = frozenset((
    "completed", "failed", "timeout", "cancelled",
))
_EXECUTION_TRANSITIONS = {
    "started": frozenset(("running", "completed", "failed", "timeout", "cancelled")),
    "running": frozenset(("completed", "failed", "timeout", "cancelled")),
}


def _canonical_actor(actor: str | None) -> str:
    """Normalize legacy ``<role>-agent`` labels before exact comparison."""
    value = str(actor or "").strip().lower()
    if value.endswith("-agent"):
        return value[:-len("-agent")]
    return value


class RouteConflictError(Exception):
    """Raised when a CAS operation detects a version conflict."""


class ControllerModelError(Exception):
    """Raised when a controller model is not permitted by policy."""


class WorkflowStateError(Exception):
    """Raised when a workflow phase transition is invalid."""


class WorkflowPhaseStateError(Exception):
    """Raised when a workflow phase transition is invalid."""


def _utcnow_age(max_age_seconds: int) -> str:
    """Return an ISO-8601 timestamp that is *max_age_seconds* in the past."""
    cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(seconds=max_age_seconds)
    return cutoff.isoformat()

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
    fallback_models_json TEXT NOT NULL DEFAULT '[]',
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
    released_at TEXT,
    configuration_hash TEXT,
    routing_mode TEXT NOT NULL DEFAULT 'fixed',
    deployment_group TEXT,
    allowed_deployments_json TEXT,
    deployment_policy_digest TEXT,
    provider_ids_json TEXT
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

CREATE TABLE IF NOT EXISTS litellm_generations (
    generation INTEGER PRIMARY KEY AUTOINCREMENT,
    registry_hash TEXT NOT NULL,
    model_count INTEGER NOT NULL,
    config_digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'staging' CHECK(status IN ('staging','active','draining','retired')),
    reason TEXT,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    retired_at TEXT
);

CREATE TABLE IF NOT EXISTS litellm_deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL REFERENCES litellm_generations(generation),
    port INTEGER NOT NULL,
    pid INTEGER,
    status TEXT NOT NULL DEFAULT 'starting' CHECK(status IN ('starting','active','draining','dead','failed')),
    health_checked_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    epoch_id TEXT NOT NULL,
    phase_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','active','completed','skipped','failed','rolled_back')),
    started_at TEXT,
    completed_at TEXT,
    actor TEXT,
    result_evidence TEXT,
    error TEXT,
    UNIQUE(run_id, epoch_id, phase_id)
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
    if current < 3:
        _migrate_v3(conn)
        conn.execute("PRAGMA user_version = 3")
    if current < 4:
        _migrate_v4(conn)
        conn.execute("PRAGMA user_version = 4")
    if current < 5:
        _migrate_v5(conn)
        conn.execute("PRAGMA user_version = 5")
    if current < 6:
        _migrate_v6(conn)
        conn.execute("PRAGMA user_version = 6")
    if current < 7:
        _migrate_v7(conn)
        conn.execute("PRAGMA user_version = 7")
    if current < 8:
        _migrate_v8(conn)
        conn.execute("PRAGMA user_version = 8")
    if current < 9:
        _migrate_v9(conn)
        conn.execute("PRAGMA user_version = 9")
    if current < 10:
        _migrate_v10(conn)
        conn.execute("PRAGMA user_version = 10")
    if current < 11:
        _migrate_v11(conn)
        conn.execute("PRAGMA user_version = 11")
    if current < 12:
        _migrate_v12(conn)
        conn.execute("PRAGMA user_version = 12")
    if current < 13:
        _migrate_v13(conn)
        conn.execute("PRAGMA user_version = 13")
    if current < 14:
        _migrate_v14(conn)
        conn.execute("PRAGMA user_version = 14")
    if current < 15:
        _migrate_v15(conn)
        conn.execute("PRAGMA user_version = 15")
    if current < 16:
        _migrate_v16(conn)
        conn.execute("PRAGMA user_version = 16")
    if current < 17:
        _migrate_v17(conn)
        conn.execute("PRAGMA user_version = 17")
    if current < 18:
        _migrate_v18(conn)
        conn.execute("PRAGMA user_version = 18")
    if current < 19:
        _migrate_v19(conn)
        conn.execute("PRAGMA user_version = 19")
    if current < 20:
        _migrate_v20(conn)
        conn.execute("PRAGMA user_version = 20")
    if current < 21:
        _migrate_v21(conn)
        conn.execute("PRAGMA user_version = 21")
    if current < 22:
        _migrate_v22(conn)
        conn.execute("PRAGMA user_version = 22")
    if current < 23:
        _migrate_v23(conn)
        conn.execute("PRAGMA user_version = 23")
    if current < 24:
        _migrate_v24(conn)
        conn.execute("PRAGMA user_version = 24")
    if current < 25:
        _migrate_v25(conn)
        conn.execute("PRAGMA user_version = 25")
    if current < 26:
        _migrate_v26(conn)
        conn.execute("PRAGMA user_version = 26")
    if current < 27:
        _migrate_v27(conn)
        conn.execute("PRAGMA user_version = 27")
    if current < 28:
        _migrate_v28(conn)
        conn.execute("PRAGMA user_version = 28")
    if current < 29:
        _migrate_v29(conn)
        conn.execute("PRAGMA user_version = 29")
    if current < 30:
        _migrate_v30(conn)
        conn.execute("PRAGMA user_version = 30")
    if current < 31:
        _migrate_v31(conn)
        conn.execute("PRAGMA user_version = 31")
    if current < 32:
        _migrate_v32(conn)
        conn.execute("PRAGMA user_version = 32")
    if current < 33:
        _migrate_v33(conn)
        conn.execute("PRAGMA user_version = 33")
    if current < 34:
        _migrate_v34(conn)
        conn.execute("PRAGMA user_version = 34")
    if current < 35:
        _migrate_v35(conn)
        conn.execute("PRAGMA user_version = 35")
    if current < 36:
        _migrate_v36(conn)
        conn.execute("PRAGMA user_version = 36")
    if current < 37 and _needs_v37_hardening(conn):
        _migrate_v37(conn)


def _needs_v37_hardening(conn: sqlite3.Connection) -> bool:
    """Detect the additive safety schema independently of user_version."""
    required = {
        "runs": {"controller_capability_hash"},
        "workspaces": {
            "canonical_generation", "current_base_sha", "current_dirty_hash",
            "last_changeset_id", "parent_canonical_generation",
            "parent_dirty_patch_hash",
        },
        "execution_changesets": {"parent_canonical_generation"},
        "agent_executions": {
            "error_class", "schema_valid", "evidence_valid", "accepted_by_controller",
            "quality_score", "verdict", "confidence",
        },
        "workflow_phases": {
            "execution_kind", "required_actor", "started_by_actor", "started_by_principal",
            "max_parallelism", "required_successes", "max_attempts", "max_attempts_per_model",
            "sidecar_id",
        },
        "runnable_action_claims": {"spawn_call_id", "execution_id", "action_kind"},
        "role_routes": {"fallback_models_json"},
    }
    for table, columns in required.items():
        present = {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not columns.issubset(present):
            return True
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='integration_journal'"
    ).fetchone() is None or conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='execution_events'"
    ).fetchone() is None or conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='completion_tokens'"
    ).fetchone() is None


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


def _migrate_v3(conn: sqlite3.Connection) -> None:
    """Add LiteLLM catalog generation and deployment tracking tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS litellm_generations (
            generation INTEGER PRIMARY KEY AUTOINCREMENT,
            registry_hash TEXT NOT NULL,
            model_count INTEGER NOT NULL,
            config_digest TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'staging' CHECK(status IN ('staging','active','draining','retired')),
            reason TEXT,
            created_at TEXT NOT NULL,
            activated_at TEXT,
            retired_at TEXT
        );

        CREATE TABLE IF NOT EXISTS litellm_deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL REFERENCES litellm_generations(generation),
            port INTEGER NOT NULL,
            pid INTEGER,
            status TEXT NOT NULL DEFAULT 'starting' CHECK(status IN ('starting','active','draining','dead','failed')),
            health_checked_at TEXT,
            created_at TEXT NOT NULL
        );
    """)


def _migrate_v4(conn: sqlite3.Connection) -> None:
    """Add api_key_env column for OpenAI-compatible backend support."""
    conn.executescript("""
        ALTER TABLE agent_bindings ADD COLUMN api_key_env TEXT;
    """)


def _migrate_v5(conn: sqlite3.Connection) -> None:
    """Add binding_versions, binding_commands, controller_policies tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS binding_versions (
            binding_id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS binding_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            claude_session_id TEXT NOT NULL DEFAULT '',
            command_type TEXT NOT NULL,
            actor_type TEXT NOT NULL DEFAULT 'controller',
            reason TEXT,
            claude_agent_id TEXT NOT NULL DEFAULT '',
            expected_binding_version INTEGER,
            requested_model_id TEXT,
            requested_role TEXT,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applied','cancelled')),
            applied_at TEXT,
            actor_id TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS controller_policies (
            run_id TEXT PRIMARY KEY,
            permitted_models TEXT NOT NULL,
            model_change_policy TEXT NOT NULL DEFAULT 'reject' CHECK(model_change_policy IN ('allow','reject')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)


def _migrate_v6(conn: sqlite3.Connection) -> None:
    """Placeholder migration (no-op)."""
    pass


def _migrate_v7(conn: sqlite3.Connection) -> None:
    """Add claude_session_id and claude_parent_agent_id to agent_bindings."""
    conn.executescript("""
        ALTER TABLE agent_bindings ADD COLUMN claude_session_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE agent_bindings ADD COLUMN claude_parent_agent_id TEXT NOT NULL DEFAULT '';
    """)
    # Recreate the unique index to include claude_session_id
    conn.execute("DROP INDEX IF EXISTS uq_active_agent_binding")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_active_agent_binding
        ON agent_bindings(run_id, claude_agent_id) WHERE released_at IS NULL
    """)


def _migrate_v8(conn: sqlite3.Connection) -> None:
    """Add CHECK constraint on binding_commands.command_type."""
    # SQLite doesn't support ALTER TABLE ... ADD CHECK easily, so we recreate the table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS binding_commands_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            claude_session_id TEXT NOT NULL DEFAULT '',
            command_type TEXT NOT NULL,
            actor_type TEXT NOT NULL DEFAULT 'controller',
            reason TEXT,
            claude_agent_id TEXT NOT NULL DEFAULT '',
            expected_binding_version INTEGER,
            requested_model_id TEXT,
            requested_role TEXT,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applied','cancelled')),
            applied_at TEXT,
            actor_id TEXT,
            created_at TEXT NOT NULL,
            CHECK(command_type IN ('model_change','profile_set','route_change','binding_release','binding_reenable'))
        )
    """)
    # Copy data from old table if it exists
    try:
        conn.execute("""
            INSERT INTO binding_commands_new
            (id, command_id, run_id, epoch_id, claude_session_id, command_type, actor_type,
             reason, claude_agent_id, expected_binding_version, requested_model_id, requested_role,
             status, applied_at, actor_id, created_at)
            SELECT id, command_id, run_id, epoch_id, claude_session_id, command_type, actor_type,
                   reason, claude_agent_id, expected_binding_version, requested_model_id, requested_role,
                   status, applied_at, actor_id, created_at
            FROM binding_commands
        """)
    except Exception:
        pass  # table didn't exist yet
    conn.execute("DROP TABLE IF EXISTS binding_commands")
    conn.execute("ALTER TABLE binding_commands_new RENAME TO binding_commands")


def _migrate_v9(conn: sqlite3.Connection) -> None:
    """v9: Create mutation_leases table for durable mutual exclusion on mutation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mutation_leases (
            run_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            role TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            released_at TEXT,
            PRIMARY KEY (run_id, agent_id)
        );
        CREATE INDEX IF NOT EXISTS idx_mutation_leases_run_active
            ON mutation_leases(run_id, released_at)
            WHERE released_at IS NULL
    """)


def _migrate_v10(conn: sqlite3.Connection) -> None:
    """Add termination_reason column to litellm_deployments.

    The base DDL may already include this column, so check before ALTER.
    """
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('litellm_deployments') WHERE name='termination_reason'"
    ).fetchone()
    if row is None:
        conn.execute("ALTER TABLE litellm_deployments ADD COLUMN termination_reason TEXT")


def _migrate_v11(conn: sqlite3.Connection) -> None:
    """v11: Create workflow_phases table for workflow phase lifecycle tracking."""

    existing = conn.execute(
        "SELECT 1 FROM pragma_table_info('workflow_phases') WHERE name='phase_id'"
    ).fetchone()
    if existing:
        return
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workflow_phases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            phase_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','active','completed','skipped','failed','rolled_back')),
            started_at TEXT,
            completed_at TEXT,
            actor TEXT,
            result_evidence TEXT,
            error TEXT,
            UNIQUE(run_id, epoch_id, phase_id)
        );
    """)


def _migrate_v12(conn: sqlite3.Connection) -> None:
    """v12: Add auth_spec_json column to agent_bindings for pinned auth."""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('agent_bindings') WHERE name='auth_spec_json'"
    ).fetchone()
    if row is None:
        conn.execute("ALTER TABLE agent_bindings ADD COLUMN auth_spec_json TEXT")


def _migrate_v13(conn: sqlite3.Connection) -> None:
    """v13: Extend workflow_phases with full phase semantics."""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('workflow_phases') WHERE name='required'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript("""
        ALTER TABLE workflow_phases ADD COLUMN required INTEGER NOT NULL DEFAULT 1;
        ALTER TABLE workflow_phases ADD COLUMN mutating INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE workflow_phases ADD COLUMN allowed_roles_json TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE workflow_phases ADD COLUMN dependencies_json TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE workflow_phases ADD COLUMN condition_json TEXT;
        ALTER TABLE workflow_phases ADD COLUMN parallel_group TEXT;
        ALTER TABLE workflow_phases ADD COLUMN specification_hash TEXT;
    """)


def _migrate_v14(conn: sqlite3.Connection) -> None:
    """v14: Create findings table for structured finding lifecycle."""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('findings') WHERE name='finding_id'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            source_phase_id TEXT,
            source_agent_id TEXT,
            severity TEXT NOT NULL DEFAULT 'medium' CHECK(severity IN ('low','medium','high','critical')),
            category TEXT DEFAULT '',
            description TEXT NOT NULL,
            evidence_json TEXT DEFAULT '{}',
            disposition TEXT CHECK(disposition IN ('pending','accepted','rejected','duplicate','waived')),
            disposition_reason TEXT,
            dispositioned_at TEXT,
            dispositioned_by TEXT,
            repair_agent_id TEXT,
            repair_phase_id TEXT,
            resolution_evidence_json TEXT,
            verification_status TEXT CHECK(verification_status IN ('pending','verified','failed','irrelevant')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_findings_epoch ON findings(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_findings_disposition ON findings(disposition);
        CREATE INDEX IF NOT EXISTS idx_findings_verification ON findings(verification_status);
    """)


def _migrate_v15(conn: sqlite3.Connection) -> None:
    """v15: Create agent_executions table for authoritative execution lifecycle."""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('agent_executions') WHERE name='execution_id'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            claude_agent_id TEXT NOT NULL,
            role TEXT NOT NULL,
            binding_id INTEGER REFERENCES agent_bindings(id),
            model_id TEXT NOT NULL,
            phase_id TEXT,
            status TEXT NOT NULL DEFAULT 'started' CHECK(status IN ('started','running','completed','failed','timeout','cancelled')),
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at TEXT,
            result_type TEXT,
            result_summary TEXT,
            output_hash TEXT,
            error TEXT,
            tool_call_count INTEGER DEFAULT 0,
            total_tokens INTEGER,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_agent_exec_run ON agent_executions(run_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_epoch ON agent_executions(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_agent ON agent_executions(claude_agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_status ON agent_executions(status);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_phase ON agent_executions(phase_id);
    """)

def _migrate_v16(conn: sqlite3.Connection) -> None:
    """v16: Create model_certification table for compatibility harness results."""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('model_certification') WHERE name='certification_id'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_certification (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            certification_id TEXT NOT NULL UNIQUE,
            model_id TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            harness_version TEXT NOT NULL,
            protocol_version TEXT NOT NULL,
            tool_call_pass INTEGER NOT NULL,
            streaming_pass INTEGER NOT NULL,
            parallel_tool_behavior TEXT,
            cancellation_pass INTEGER NOT NULL,
            max_validated_context INTEGER,
            provider_endpoint_digest TEXT,
            certified_at TEXT NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cert_model ON model_certification(model_id);
        CREATE INDEX IF NOT EXISTS idx_cert_valid ON model_certification(certified_at, expires_at);
    """)


def _migrate_v17(conn: sqlite3.Connection) -> None:
    """v17: logical-model endpoint pinning and cache observations."""
    binding_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(agent_bindings)").fetchall()
    }
    for name, sql_type in (
        ("endpoint_id", "TEXT"),
        ("endpoint_selection_reason", "TEXT"),
        ("endpoint_policy_json", "TEXT"),
        ("certification_id", "TEXT"),
        ("provider_id", "TEXT"),
    ):
        if name not in binding_columns:
            conn.execute(f"ALTER TABLE agent_bindings ADD COLUMN {name} {sql_type}")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_endpoint_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            request_id TEXT,
            input_tokens_total INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms REAL,
            succeeded INTEGER NOT NULL,
            configuration_hash TEXT NOT NULL,
            observed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_endpoint_usage_model
            ON model_endpoint_usage(model_id, endpoint_id, observed_at);

        CREATE TABLE IF NOT EXISTS model_endpoint_metrics (
            model_id TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            input_tokens_total INTEGER NOT NULL,
            cache_read_tokens INTEGER NOT NULL,
            cache_rate REAL NOT NULL,
            median_latency_ms REAL,
            success_rate REAL,
            calculated_at TEXT NOT NULL,
            PRIMARY KEY (model_id, endpoint_id, configuration_hash)
        );

        CREATE TABLE IF NOT EXISTS controller_bindings (
            binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            client_session_id TEXT NOT NULL,
            public_model TEXT NOT NULL,
            registry_model_id TEXT NOT NULL,
            backend TEXT NOT NULL,
            upstream_model TEXT,
            provider_id TEXT,
            api_base TEXT,
            catalog_generation INTEGER,
            registry_hash TEXT NOT NULL,
            certification_id TEXT,
            auth_spec_json TEXT,
            api_key_env TEXT,
            endpoint_id TEXT,
            provider_ids_json TEXT,
            endpoint_selection_reason TEXT,
            endpoint_policy_json TEXT,
            bound_at TEXT NOT NULL,
            released_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_active_controller_binding
            ON controller_bindings(run_id, client_session_id)
            WHERE released_at IS NULL;
    """)


def _migrate_v18(conn: sqlite3.Connection) -> None:
    """v18: persist the complete immutable workflow phase contract."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(workflow_phases)").fetchall()}
    additions = {
        "ordinal": "INTEGER",
        "distinct_agent_from_json": "TEXT NOT NULL DEFAULT '[]'",
        "max_duration_seconds": "INTEGER",
        "turn_budget": "INTEGER",
        "provider_requirements_json": "TEXT NOT NULL DEFAULT '[]'",
        "min_fanout": "INTEGER NOT NULL DEFAULT 1",
        "max_fanout": "INTEGER NOT NULL DEFAULT 1",
        "result_schema": "TEXT",
        "quality_quorum": "INTEGER NOT NULL DEFAULT 1",
        "fallback_policy": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE workflow_phases ADD COLUMN {name} {declaration}")


def _migrate_v19(conn: sqlite3.Connection) -> None:
    """Repair the v15 agent-executions foreign key and verify the database."""
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_executions'"
    ).fetchone()
    if table is None:
        return

    foreign_keys = conn.execute("PRAGMA foreign_key_list(agent_executions)").fetchall()
    malformed = any(
        row[2] == "agent_bindings" and row[4] != "binding_id"
        for row in foreign_keys
    )
    if malformed:
        conn.execute("ALTER TABLE agent_executions RENAME TO agent_executions_malformed")
        conn.executescript("""
            CREATE TABLE agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                epoch_id TEXT NOT NULL,
                claude_agent_id TEXT NOT NULL,
                role TEXT NOT NULL,
                binding_id INTEGER REFERENCES agent_bindings(binding_id),
                model_id TEXT NOT NULL,
                phase_id TEXT,
                status TEXT NOT NULL DEFAULT 'started' CHECK(status IN ('started','running','completed','failed','timeout','cancelled')),
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                result_type TEXT,
                result_summary TEXT,
                output_hash TEXT,
                error TEXT,
                tool_call_count INTEGER DEFAULT 0,
                total_tokens INTEGER,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO agent_executions
                (id, execution_id, run_id, epoch_id, claude_agent_id, role, binding_id,
                 model_id, phase_id, status, started_at, completed_at, result_type,
                 result_summary, output_hash, error, tool_call_count, total_tokens, updated_at)
            SELECT id, execution_id, run_id, epoch_id, claude_agent_id, role, binding_id,
                   model_id, phase_id, status, started_at, completed_at, result_type,
                   result_summary, output_hash, error, tool_call_count, total_tokens, updated_at
            FROM agent_executions_malformed;
            DROP TABLE agent_executions_malformed;
        """)

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_agent_exec_run ON agent_executions(run_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_epoch ON agent_executions(epoch_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_agent ON agent_executions(claude_agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_status ON agent_executions(status);
        CREATE INDEX IF NOT EXISTS idx_agent_exec_phase ON agent_executions(phase_id);
    """)
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(
            f"foreign-key violations remain after v19 migration: {violations!r}"
        )


def _migrate_v20(conn: sqlite3.Connection) -> None:
    """Store endpoint/configuration-specific compatibility certifications."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS endpoint_certifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            certification_id TEXT NOT NULL UNIQUE,
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            litellm_version TEXT,
            harness_version TEXT NOT NULL,
            protocol_version TEXT NOT NULL,
            capability TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pass','fail','expired')),
            certified_at TEXT NOT NULL,
            expires_at TEXT,
            evidence_digest TEXT NOT NULL,
            UNIQUE(provider_id, model_id, endpoint_id, configuration_hash, capability)
        );
        CREATE INDEX IF NOT EXISTS idx_endpoint_cert_lookup
            ON endpoint_certifications(provider_id, model_id, endpoint_id, configuration_hash, status);
    """)


def _migrate_v21(conn: sqlite3.Connection) -> None:
    """Pin endpoint-qualified LiteLLM aliases for controller bindings."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(controller_bindings)").fetchall()}
    if "litellm_model_name" not in columns:
        conn.execute("ALTER TABLE controller_bindings ADD COLUMN litellm_model_name TEXT")
    if "configuration_hash" not in columns:
        conn.execute("ALTER TABLE controller_bindings ADD COLUMN configuration_hash TEXT")


def _migrate_v22(conn: sqlite3.Connection) -> None:
    """Persist task intake facts without storing the full user prompt."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS task_intakes (
            intake_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            session_id TEXT NOT NULL,
            prompt_digest TEXT NOT NULL,
            request_kind TEXT NOT NULL,
            repository_features_json TEXT NOT NULL,
            deterministic_signals_json TEXT NOT NULL,
            minimum_tier TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_intakes_run ON task_intakes(run_id, created_at);
    """)


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, name: str, declaration: str
) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _migrate_v23(conn: sqlite3.Connection) -> None:
    """Persist fastpath recommendations and authoritative dispositions."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS route_proposals (
            proposal_id TEXT PRIMARY KEY,
            intake_id TEXT NOT NULL REFERENCES task_intakes(intake_id),
            source TEXT NOT NULL,
            fastpath_model_id TEXT,
            fastpath_endpoint_id TEXT,
            raw_output_digest TEXT,
            parsed_proposal_json TEXT NOT NULL,
            confidence REAL,
            validation_status TEXT NOT NULL,
            validation_reason TEXT,
            controller_disposition TEXT,
            controller_reason TEXT,
            applied_epoch_id TEXT,
            created_at TEXT NOT NULL,
            applied_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_route_proposals_intake
            ON route_proposals(intake_id, created_at);

        CREATE TABLE IF NOT EXISTS fastpath_verifications (
            verification_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            phase_id TEXT,
            workspace_fingerprint TEXT,
            contract_digest TEXT NOT NULL,
            evidence_digest TEXT NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('pass','fail','escalate')),
            checks_json TEXT NOT NULL,
            violations_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            policy_disposition TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fastpath_verify_epoch
            ON fastpath_verifications(run_id, epoch_id, created_at);
    """)


def _migrate_v24(conn: sqlite3.Connection) -> None:
    """Add durable execution metrics and provider agent reservations."""
    for name, declaration in {
        "provider_id": "TEXT",
        "endpoint_id": "TEXT",
        "transport": "TEXT",
        "configuration_hash": "TEXT",
        "litellm_generation": "INTEGER",
        "request_count": "INTEGER NOT NULL DEFAULT 0",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "input_tokens": "INTEGER NOT NULL DEFAULT 0",
        "cache_read_tokens": "INTEGER NOT NULL DEFAULT 0",
        "cache_write_tokens": "INTEGER NOT NULL DEFAULT 0",
        "output_tokens": "INTEGER NOT NULL DEFAULT 0",
        "ttft_ms": "REAL",
        "wall_time_ms": "REAL",
        "rate_limit_count": "INTEGER NOT NULL DEFAULT 0",
        "upstream_5xx_count": "INTEGER NOT NULL DEFAULT 0",
        "tool_parse_failures": "INTEGER NOT NULL DEFAULT 0",
        "controller_interventions": "INTEGER NOT NULL DEFAULT 0",
        "parent_execution_id": "TEXT",
        "execution_kind": "TEXT NOT NULL DEFAULT 'subagent'",
        "actor_kind": "TEXT NOT NULL DEFAULT 'subagent'",
        "workspace_id": "TEXT",
        "independence_key": "TEXT",
        "result_json": "TEXT",
    }.items():
        _add_column_if_missing(conn, "agent_executions", name, declaration)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS provider_reservations (
            reservation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT,
            provider_id TEXT NOT NULL,
            execution_id TEXT,
            lane TEXT NOT NULL DEFAULT 'worker',
            state TEXT NOT NULL CHECK(state IN ('queued','reserved','released','expired','cancelled')),
            queued_at TEXT NOT NULL,
            admitted_at TEXT,
            released_at TEXT,
            deadline_at TEXT,
            reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_provider_reservations_active
            ON provider_reservations(provider_id, state, queued_at);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_active_execution
            ON provider_reservations(execution_id)
            WHERE execution_id IS NOT NULL AND state IN ('queued','reserved');
    """)


def _migrate_v25(conn: sqlite3.Connection) -> None:
    """Persist spawn intents, request attempts, quota and circuit state."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS spawn_intents (
            intent_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            phase_id TEXT,
            native_agent_name TEXT NOT NULL,
            role TEXT NOT NULL,
            slot TEXT NOT NULL DEFAULT 'inherit',
            model_id TEXT,
            endpoint_id TEXT,
            provider_id TEXT,
            execution_kind TEXT NOT NULL DEFAULT 'subagent',
            status TEXT NOT NULL CHECK(status IN ('planned','queued','spawned','completed','failed','cancelled')),
            created_at TEXT NOT NULL,
            spawned_at TEXT,
            completed_at TEXT,
            fallback_of TEXT,
            policy_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_spawn_intents_epoch
            ON spawn_intents(run_id, epoch_id, status);

        CREATE TABLE IF NOT EXISTS request_attempts (
            request_id TEXT PRIMARY KEY,
            execution_id TEXT,
            logical_model_id TEXT NOT NULL,
            endpoint_id TEXT,
            provider_id TEXT,
            provider_model TEXT,
            attempt INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL,
            queued_at TEXT,
            admitted_at TEXT,
            started_at TEXT,
            first_token_at TEXT,
            completed_at TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            status_code INTEGER,
            error_class TEXT,
            cancelled INTEGER NOT NULL DEFAULT 0,
            provider_request_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_request_attempts_execution
            ON request_attempts(execution_id, attempt);

        CREATE TABLE IF NOT EXISTS provider_quota_state (
            provider_id TEXT PRIMARY KEY,
            window_started_at TEXT,
            requests_minute INTEGER NOT NULL DEFAULT 0,
            requests_day INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS endpoint_circuits (
            provider_id TEXT NOT NULL,
            endpoint_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('healthy','degraded','open','half-open')),
            open_until TEXT,
            failure_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(provider_id, endpoint_id)
        );

        CREATE TABLE IF NOT EXISTS provider_catalog_entries (
            provider_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            endpoint_id TEXT,
            availability TEXT NOT NULL DEFAULT 'unknown',
            raw_json TEXT NOT NULL,
            discovered_at TEXT NOT NULL,
            response_digest TEXT NOT NULL,
            PRIMARY KEY(provider_id, model_id, endpoint_id)
        );
    """)


def _migrate_v26(conn: sqlite3.Connection) -> None:
    """Persist isolated workspaces and changesets for parallel workers."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('main','shadow','integration')),
            path TEXT NOT NULL,
            base_sha TEXT,
            dirty_patch_hash TEXT,
            status TEXT NOT NULL CHECK(status IN ('active','ready','merged','discarded','failed')),
            owner_execution_id TEXT,
            created_at TEXT NOT NULL,
            released_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_workspaces_epoch
            ON workspaces(run_id, epoch_id, status);

        CREATE TABLE IF NOT EXISTS execution_changesets (
            changeset_id TEXT PRIMARY KEY,
            execution_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            base_sha TEXT NOT NULL,
            patch_digest TEXT NOT NULL,
            changed_files_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('proposed','validated','rejected','merged')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS integration_candidates (
            candidate_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            changeset_id TEXT NOT NULL REFERENCES execution_changesets(changeset_id),
            overlap_json TEXT NOT NULL,
            validation_json TEXT NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('green','yellow','red','pending')),
            integration_execution_id TEXT,
            created_at TEXT NOT NULL
        );
    """)


def _migrate_v27(conn: sqlite3.Connection) -> None:
    """Extend epochs and routes with immutable task/endpoint identity."""
    for name, declaration in {
        "prompt_digest": "TEXT",
        "minimum_tier": "TEXT",
        "intake_id": "TEXT",
        "contract_json": "TEXT",
        "baseline_fingerprint": "TEXT",
        "controller_binding_id": "INTEGER",
    }.items():
        _add_column_if_missing(conn, "epochs", name, declaration)
    for name, declaration in {
        "endpoint_id": "TEXT",
        "endpoint_override": "TEXT",
        "selection_reason": "TEXT",
    }.items():
        _add_column_if_missing(conn, "role_routes", name, declaration)


def _migrate_v28(conn: sqlite3.Connection) -> None:
    """Scope mutation leases to canonical workspaces, not merely runs."""
    _add_column_if_missing(conn, "mutation_leases", "workspace_id", "TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_mutator_workspace "
        "ON mutation_leases(workspace_id) WHERE released_at IS NULL AND workspace_id IS NOT NULL"
    )


def _migrate_v29(conn: sqlite3.Connection) -> None:
    """Persist immutable provider catalog responses and their generations."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS provider_catalog_snapshots (
            generation INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_id TEXT NOT NULL,
            response_digest TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            request_id TEXT,
            models_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_catalog_snapshots_provider
            ON provider_catalog_snapshots(provider_id, fetched_at);
    """)


def _migrate_v30(conn: sqlite3.Connection) -> None:
    """Add endpoint configuration identity to historical agent bindings."""
    _add_column_if_missing(conn, "agent_bindings", "configuration_hash", "TEXT")


def _migrate_v31(conn: sqlite3.Connection) -> None:
    """Persist logical deployment-group binding policy."""
    for name, declaration in {
        "routing_mode": "TEXT NOT NULL DEFAULT 'fixed'",
        "deployment_group": "TEXT",
        "allowed_deployments_json": "TEXT",
        "deployment_policy_digest": "TEXT",
    }.items():
        _add_column_if_missing(conn, "agent_bindings", name, declaration)
    for name, declaration in {
        "routing_mode": "TEXT NOT NULL DEFAULT 'fixed'",
        "deployment_group": "TEXT",
        "allowed_deployments_json": "TEXT",
        "deployment_policy_digest": "TEXT",
    }.items():
        _add_column_if_missing(conn, "controller_bindings", name, declaration)


def _migrate_v32(conn: sqlite3.Connection) -> None:
    """Persist the untracked-file portion of a shadow baseline."""
    _add_column_if_missing(
        conn, "workspaces", "baseline_untracked_json", "TEXT NOT NULL DEFAULT '[]'"
    )


def _migrate_v33(conn: sqlite3.Connection) -> None:
    """Persist worker patches so yellow candidates remain integrable."""
    _add_column_if_missing(conn, "execution_changesets", "patch_blob", "BLOB")


def _migrate_v34(conn: sqlite3.Connection) -> None:
    """Allow a controller to close a red/yellow integration decision."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(integration_candidates)").fetchall()}
    if not columns:
        return
    conn.execute("ALTER TABLE integration_candidates RENAME TO integration_candidates_v34_old")
    conn.executescript("""
        CREATE TABLE integration_candidates (
            candidate_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            changeset_id TEXT NOT NULL REFERENCES execution_changesets(changeset_id),
            overlap_json TEXT NOT NULL,
            validation_json TEXT NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('green','yellow','red','pending','resolved')),
            integration_execution_id TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO integration_candidates
            (candidate_id, run_id, epoch_id, changeset_id, overlap_json,
             validation_json, disposition, integration_execution_id, created_at)
        SELECT candidate_id, run_id, epoch_id, changeset_id, overlap_json,
               validation_json, disposition, integration_execution_id, created_at
        FROM integration_candidates_v34_old;
        DROP TABLE integration_candidates_v34_old;
    """)


def _migrate_v35(conn: sqlite3.Connection) -> None:
    """Persist cooperative native-agent action claims.

    Claude Code owns the native ``Agent`` process and cannot replay a tool
    call that a hook denied.  A claim is therefore created by the controller
    before spawning, then consumed exactly once by ``PreToolUse Agent``.
    """
    _add_column_if_missing(conn, "spawn_intents", "claude_agent_id", "TEXT")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runnable_action_claims (
            action_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            phase_id TEXT NOT NULL,
            role TEXT NOT NULL,
            native_agent_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider_id TEXT,
            claim_token TEXT NOT NULL UNIQUE,
            reservation_id TEXT,
            intent_id TEXT,
            status TEXT NOT NULL CHECK(status IN ('claimed','consumed','expired','cancelled')),
            created_at TEXT NOT NULL,
            claimed_at TEXT NOT NULL,
            consumed_at TEXT,
            expires_at TEXT NOT NULL,
            claude_agent_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_action_claims_spawn
            ON runnable_action_claims(run_id, epoch_id, native_agent_name, status);
        CREATE INDEX IF NOT EXISTS idx_action_claims_agent
            ON runnable_action_claims(run_id, claude_agent_id, status);
    """)


def _migrate_v36(conn: sqlite3.Connection) -> None:
    """Persist all provider candidates for managed deployment groups."""
    _add_column_if_missing(conn, "agent_bindings", "provider_ids_json", "TEXT")
    _add_column_if_missing(conn, "controller_bindings", "provider_ids_json", "TEXT")


def _migrate_v37(conn: sqlite3.Connection) -> None:
    """Harden actor credentials, native spawn correlation, and workspace generations."""
    _add_column_if_missing(conn, "runs", "controller_capability_hash", "TEXT")
    for name, declaration in {
        "canonical_generation": "INTEGER NOT NULL DEFAULT 0",
        "current_base_sha": "TEXT",
        "current_dirty_hash": "TEXT",
        "last_changeset_id": "TEXT",
        "parent_canonical_generation": "INTEGER",
        "parent_dirty_patch_hash": "TEXT",
    }.items():
        _add_column_if_missing(conn, "workspaces", name, declaration)
    _add_column_if_missing(
        conn, "execution_changesets", "parent_canonical_generation", "INTEGER"
    )
    _add_column_if_missing(conn, "agent_executions", "error_class", "TEXT")
    for name, declaration in {
        "schema_valid": "INTEGER",
        "evidence_valid": "INTEGER",
        "accepted_by_controller": "INTEGER",
        "quality_score": "REAL",
        "verdict": "TEXT",
        "confidence": "REAL",
    }.items():
        _add_column_if_missing(conn, "agent_executions", name, declaration)
    _add_column_if_missing(
        conn, "workflow_phases", "execution_kind",
        "TEXT NOT NULL DEFAULT 'native_agent'",
    )
    _add_column_if_missing(conn, "workflow_phases", "required_actor", "TEXT")
    _add_column_if_missing(conn, "workflow_phases", "started_by_actor", "TEXT")
    _add_column_if_missing(conn, "workflow_phases", "started_by_principal", "TEXT")
    for name, declaration in {
        "max_parallelism": "INTEGER",
        "required_successes": "INTEGER",
        "max_attempts": "INTEGER NOT NULL DEFAULT 1",
        "max_attempts_per_model": "INTEGER",
        "sidecar_id": "TEXT",
    }.items():
        _add_column_if_missing(conn, "workflow_phases", name, declaration)
    _add_column_if_missing(conn, "role_routes", "fallback_models_json", "TEXT NOT NULL DEFAULT '[]'")
    conn.execute(
        "UPDATE workflow_phases SET required_actor=actor "
        "WHERE required_actor IS NULL"
    )
    conn.execute(
        "UPDATE workspaces SET current_base_sha=COALESCE(current_base_sha, base_sha), "
        "current_dirty_hash=COALESCE(current_dirty_hash, dirty_patch_hash)"
    )

    # v35 only allowed ``claimed`` and ``consumed``.  The latter was being
    # used as both a transient spawn state and a successful terminal state,
    # which made reconciliation impossible.  Rebuild the small ledger with
    # explicit terminal outcomes and deterministic spawn correlation fields.
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(runnable_action_claims)").fetchall()
    }
    if columns:
        conn.execute("DROP INDEX IF EXISTS idx_action_claims_spawn")
        conn.execute("DROP INDEX IF EXISTS idx_action_claims_agent")
        conn.execute("DROP INDEX IF EXISTS uq_pending_action_spawn")
        conn.execute("ALTER TABLE runnable_action_claims RENAME TO runnable_action_claims_v37_old")
        conn.executescript("""
            CREATE TABLE runnable_action_claims (
                action_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                epoch_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                role TEXT NOT NULL,
                native_agent_name TEXT NOT NULL,
                model_id TEXT NOT NULL,
                action_kind TEXT NOT NULL DEFAULT 'native_agent',
                provider_id TEXT,
                claim_token TEXT NOT NULL UNIQUE,
                reservation_id TEXT,
                intent_id TEXT,
                status TEXT NOT NULL CHECK(status IN (
                    'claimed','consumed','completed','failed','timed_out',
                    'cancelled','expired','orphaned'
                )),
                created_at TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                consumed_at TEXT,
                expires_at TEXT NOT NULL,
                claude_agent_id TEXT,
                spawn_call_id TEXT,
                execution_id TEXT
            );
            CREATE INDEX idx_action_claims_spawn
                ON runnable_action_claims(run_id, epoch_id, native_agent_name, status);
            CREATE INDEX idx_action_claims_agent
                ON runnable_action_claims(run_id, claude_agent_id, status);
        """)
        old_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(runnable_action_claims_v37_old)").fetchall()
        }
        extra_select = "spawn_call_id" if "spawn_call_id" in old_columns else "NULL"
        execution_select = "execution_id" if "execution_id" in old_columns else "NULL"
        conn.execute(
            "INSERT INTO runnable_action_claims "
            "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, action_kind, "
            "provider_id, claim_token, reservation_id, intent_id, status, created_at, "
            "claimed_at, consumed_at, expires_at, claude_agent_id, spawn_call_id, execution_id) "
            "SELECT action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
            "CASE WHEN action_id LIKE 'integration:%' THEN 'controller_integration' "
            "ELSE 'native_agent' END, provider_id, claim_token, reservation_id, intent_id, status, created_at, "
            "claimed_at, consumed_at, expires_at, claude_agent_id, "
            f"{extra_select}, {execution_select} FROM runnable_action_claims_v37_old"
        )
        conn.execute("DROP TABLE runnable_action_claims_v37_old")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS integration_journal (
            journal_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            changeset_id TEXT NOT NULL REFERENCES execution_changesets(changeset_id),
            expected_generation INTEGER NOT NULL,
            expected_dirty_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('applying','completed','reconciled','failed')),
            error TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_integration_journal_pending
            ON integration_journal(status, created_at);

        CREATE TABLE IF NOT EXISTS execution_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT NOT NULL REFERENCES agent_executions(execution_id),
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(execution_id, seq)
        );
        CREATE INDEX IF NOT EXISTS idx_execution_events_scope
            ON execution_events(run_id, epoch_id, execution_id, seq);

        CREATE TABLE IF NOT EXISTS completion_tokens (
            token_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            workspace_fingerprint TEXT NOT NULL,
            route_snapshot_sha256 TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            consumed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_completion_tokens_scope
            ON completion_tokens(run_id, epoch_id, consumed_at, expires_at);
    """)


# ------------------------------------------------------------------ RouteState

class RouteState:
    """SQLite-backed persistent state for runs, epochs, routes, bindings, health."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        connection_factory: Callable[[Path], sqlite3.Connection] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._connection_factory = connection_factory or self._open_sqlite
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._new_conn()
        try:
            _apply_migrations(conn)
            conn.commit()
        finally:
            conn.close()

    # ---- helpers ---------------------------------------------------

    @staticmethod
    def _open_sqlite(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _new_conn(self) -> sqlite3.Connection:
        return self._connection_factory(self.db_path)

    # ---- Run lifecycle ---------------------------------------------

    def create_task_intake(
        self,
        *,
        intake_id: str,
        run_id: str,
        session_id: str,
        prompt: str,
        request_kind: str,
        repository_features: dict,
        deterministic_signals: list[str],
        minimum_tier: str,
    ) -> dict:
        """Persist bounded intake facts before any workflow epoch is created."""
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO task_intakes "
                "(intake_id, run_id, session_id, prompt_digest, request_kind,"
                " repository_features_json, deterministic_signals_json, minimum_tier, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    intake_id,
                    run_id,
                    session_id,
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    request_kind,
                    json.dumps(repository_features, sort_keys=True, separators=(",", ":")),
                    json.dumps(sorted(set(deterministic_signals)), separators=(",", ":")),
                    minimum_tier,
                    _utcnow(),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM task_intakes WHERE intake_id=?", (intake_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def create_route_proposal(
        self,
        *,
        proposal_id: str,
        intake_id: str,
        source: str,
        parsed_proposal: dict,
        validation_status: str,
        validation_reason: str = "",
        fastpath_model_id: str | None = None,
        fastpath_endpoint_id: str | None = None,
        raw_output_digest: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO route_proposals "
                "(proposal_id, intake_id, source, fastpath_model_id, fastpath_endpoint_id,"
                " raw_output_digest, parsed_proposal_json, confidence, validation_status,"
                " validation_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (proposal_id, intake_id, source, fastpath_model_id, fastpath_endpoint_id,
                 raw_output_digest, json.dumps(parsed_proposal, sort_keys=True, separators=(",", ":")),
                 confidence, validation_status, validation_reason, _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_route_proposal(self, proposal_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_task_intake(self, intake_id: str) -> dict | None:
        """Return one bounded task-intake record for control-plane checks."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM task_intakes WHERE intake_id=?",
                (intake_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_route_proposal_for_run(
        self, proposal_id: str, run_id: str,
    ) -> dict | None:
        """Return a proposal only when its intake belongs to *run_id*."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT rp.* FROM route_proposals AS rp "
                "JOIN task_intakes AS ti ON ti.intake_id=rp.intake_id "
                "WHERE rp.proposal_id=? AND ti.run_id=?",
                (proposal_id, run_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def set_route_proposal_disposition(
        self, proposal_id: str, disposition: str, reason: str = "", epoch_id: str | None = None
    ) -> dict | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE route_proposals SET controller_disposition=?, controller_reason=?,"
                " applied_epoch_id=COALESCE(?, applied_epoch_id), applied_at=COALESCE(?, applied_at)"
                " WHERE proposal_id=?",
                (disposition, reason, epoch_id, _utcnow() if epoch_id else None, proposal_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_fastpath_verification(self, **values: object) -> dict:
        required = {
            "verification_id", "run_id", "epoch_id", "contract_digest", "evidence_digest",
            "decision", "checks_json", "violations_json", "confidence", "policy_disposition",
        }
        missing = required - values.keys()
        if missing:
            raise ValueError(f"missing fastpath verification fields: {sorted(missing)}")
        conn = self._new_conn()
        try:
            columns = sorted(values)
            placeholders = ",".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO fastpath_verifications ({','.join(columns)}, created_at) "
                f"VALUES ({placeholders}, ?)",
                [values[column] for column in columns] + [_utcnow()],
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM fastpath_verifications WHERE verification_id=?",
                (values["verification_id"],),
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

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
    ) -> dict:
        """Reserve a provider-wide native-agent slot durably.

        This complements the in-process request admission manager.  A queued
        native agent is not spawned until this record becomes ``reserved``.
        """
        if max_active < 1:
            raise ValueError("max_active must be positive")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM provider_reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if existing:
                conn.commit()
                return dict(existing)
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            if int(active) >= max_active and not enqueue:
                conn.rollback()
                return {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "epoch_id": epoch_id,
                    "provider_id": provider_id,
                    "execution_id": execution_id,
                    "lane": lane,
                    "state": "unavailable",
                    "reason": reason,
                }
            status = "reserved" if int(active) < max_active else "queued"
            now = _utcnow()
            conn.execute(
                "INSERT INTO provider_reservations (reservation_id, run_id, epoch_id, provider_id, execution_id,"
                " lane, state, queued_at, admitted_at, deadline_at, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, run_id, epoch_id, provider_id, execution_id, lane, status, now,
                 now if status == "reserved" else None, deadline_at, reason),
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

    def provider_agent_capacity_available(self, provider_id: str, max_active: int) -> bool:
        """Return whether a native Agent may be spawned immediately."""
        conn = self._new_conn()
        try:
            active = conn.execute(
                "SELECT COUNT(*) FROM provider_reservations "
                "WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0]
            return int(active) < max_active
        finally:
            conn.close()

    def _active_action_claims(
        self, run_id: str, epoch_id: str,
    ) -> dict[str, dict]:
        """Return non-terminal action claims, expiring stale claims first.

        Expiring the claim row alone is not enough: a TTL-expired claim may
        still hold a 'reserved' provider_reservations row.  Without releasing
        it here, that capacity slot leaks for the life of the process --
        reconcile_lifecycle only runs at startup, so it would not otherwise
        be freed until the next restart.
        """
        conn = self._new_conn()
        try:
            now = _utcnow()
            expiring = conn.execute(
                "SELECT action_id, reservation_id, provider_id FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status='claimed' AND expires_at < ?",
                (run_id, epoch_id, now),
            ).fetchall()
            conn.execute(
                "UPDATE runnable_action_claims SET status='expired' "
                "WHERE run_id=? AND epoch_id=? AND status='claimed' AND expires_at < ?",
                (run_id, epoch_id, now),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status IN ('claimed','consumed')",
                (run_id, epoch_id),
            ).fetchall()
            result = {str(row["action_id"]): dict(row) for row in rows}
        finally:
            conn.close()
        released_provider_ids: set[str] = set()
        for row in expiring:
            reservation_id = row["reservation_id"]
            if not reservation_id:
                continue
            released = self.release_provider_reservation(str(reservation_id), "expired")
            if released and released.get("provider_id"):
                released_provider_ids.add(str(released["provider_id"]))
        if released_provider_ids:
            from enhanced_router.registry import get_registry

            registry = get_registry()
            for provider_id in released_provider_ids:
                provider = registry.providers.get(provider_id)
                if provider:
                    self.admit_provider_agents(provider_id, provider.limits.max_active_agents)
        return result

    def get_runnable_actions(
        self, run_id: str, epoch_id: str, *, include_claimed: bool = False,
    ) -> list[dict]:
        """Return controller-visible native actions that may be spawned now.

        This is planning, not execution.  It intentionally never creates a
        queued reservation: a denied Claude Code Agent call cannot be replayed
        by SQLite.  The controller asks again after a terminal lifecycle event.
        """
        from enhanced_router.registry import get_registry

        self.advance_conditional_phases(run_id, epoch_id)
        registry = get_registry()
        actions: list[dict] = []
        claims = self._active_action_claims(run_id, epoch_id)
        for candidate in self.get_integration_candidates(run_id=run_id, epoch_id=epoch_id):
            disposition = str(candidate.get("disposition"))
            if disposition not in {"yellow", "red", "pending"}:
                continue
            action_id = f"integration:{candidate['candidate_id']}"
            claim = claims.get(action_id)
            if claim is not None and not include_claimed:
                continue
            controller_model = "controller"
            run = self.get_run(run_id)
            if run and run.get("claude_session_id"):
                binding = self.get_controller_binding(
                    run_id, str(run["claude_session_id"])
                )
                if binding and binding.get("registry_model_id"):
                    controller_model = str(binding["registry_model_id"])
            action = {
                "action_id": action_id,
                "action_kind": "controller_integration",
                "requires_main_controller": True,
                "candidate_id": candidate["candidate_id"],
                "changeset_id": candidate["changeset_id"],
                "phase_id": "controller-integration",
                "role": "controller",
                "native_agent_name": "controller",
                "model_id": controller_model,
                "disposition": disposition,
                "required_action": (
                    "inspect and resolve the conflict; approve only after deterministic preflight"
                ),
                "status": "escalated",
            }
            if claim is not None:
                action.update({
                    "status": str(claim["status"]),
                    "claim_token": claim["claim_token"],
                    "reservation_id": claim.get("reservation_id"),
                    "intent_id": claim.get("intent_id"),
                })
            actions.append(action)
        for phase in self.get_ready_phases(run_id, epoch_id) + self.get_active_phases(run_id, epoch_id):
            actor_contract = str(phase.get("required_actor") or phase.get("actor") or "")
            controller_phase = actor_contract == "controller"
            if actor_contract and not controller_phase:
                continue
            if controller_phase and phase.get("status") != "active":
                continue
            allowed_roles = ["controller"] if controller_phase else json.loads(
                phase.get("allowed_roles_json") or "[]"
            )
            executions = self.get_agent_executions(
                run_id, epoch_id=epoch_id, phase_id=phase["phase_id"]
            )
            active_count = sum(
                item.get("status") in {"started", "running", "streaming", "verifying"}
                for item in executions
            )
            max_parallelism = int(
                phase.get("max_parallelism") or phase.get("max_fanout") or 1
            )
            max_attempts = int(
                phase.get("max_attempts") or phase.get("max_fanout") or 1
            )
            max_attempts_per_model = phase.get("max_attempts_per_model")
            fallback_policy = str(phase.get("fallback_policy") or "").strip().lower()
            if active_count >= max_parallelism or len(executions) >= max_attempts:
                continue
            execution_kind = str(phase.get("execution_kind") or "native_agent")
            sidecar_id = str(phase.get("sidecar_id") or "").strip() or None
            sidecar_spec = None
            if execution_kind == "sidecar_call" and sidecar_id:
                try:
                    sidecar_spec = registry.get_sidecar(sidecar_id)
                except KeyError:
                    continue
                if not sidecar_spec.enabled:
                    continue
            for role in allowed_roles:
                controller_binding = None
                if controller_phase:
                    run = self.get_run(run_id)
                    session_id = str(run.get("claude_session_id")) if run and run.get("claude_session_id") else ""
                    controller_binding = self.get_controller_binding(run_id, session_id) if session_id else None
                    if not controller_binding or not controller_binding.get("registry_model_id"):
                        continue
                    route = {
                        "endpoint_override": controller_binding.get("endpoint_id"),
                        "model_id": controller_binding["registry_model_id"],
                    }
                elif sidecar_spec is not None:
                    route = {
                        "endpoint_override": (
                            None if sidecar_spec.endpoint == "auto" else sidecar_spec.endpoint
                        ),
                        "model_id": sidecar_spec.model_id,
                        "version": 0,
                    }
                else:
                    route = self.get_role_route(run_id, epoch_id, role)
                    if not route:
                        continue
                if executions and any(
                    item.get("status") in {"failed", "timeout", "timed_out", "cancelled", "orphaned"}
                    for item in executions
                ) and not fallback_policy:
                    try:
                        route_fallbacks = json.loads(route.get("fallback_models_json") or "[]")
                    except (TypeError, ValueError):
                        route_fallbacks = []
                    if not (
                        execution_kind == "native_agent"
                        and isinstance(route_fallbacks, list)
                        and route_fallbacks
                    ):
                        continue
                if (
                    execution_kind == "native_agent"
                    and not controller_phase
                ):
                    try:
                        fallback_models = json.loads(route.get("fallback_models_json") or "[]")
                    except (TypeError, ValueError):
                        fallback_models = []
                    failed_attempts = sum(
                        item.get("status") in {"failed", "timeout", "timed_out", "cancelled", "orphaned"}
                        for item in executions
                    )
                    if fallback_models and failed_attempts > len(fallback_models):
                        continue
                    fallback_index = int(failed_attempts) - 1
                    if failed_attempts > 0 and isinstance(fallback_models, list) and fallback_index < len(fallback_models):
                        fallback_model = fallback_models[fallback_index]
                        if isinstance(fallback_model, str) and fallback_model:
                            route = dict(route)
                            route["model_id"] = fallback_model
                            route["endpoint_override"] = None
                model_id = str(route["model_id"])
                if max_attempts_per_model is not None:
                    model_attempts = sum(
                        item.get("model_id") == model_id for item in executions
                    )
                    if model_attempts >= int(max_attempts_per_model):
                        continue
                model = registry.get_model(model_id)
                provider_value = (
                    (controller_binding or {}).get("provider_id") or model.provider_id
                ) if model else None
                provider_id = str(provider_value) if provider_value else None
                provider = registry.providers.get(provider_id) if provider_id else None
                if controller_phase:
                    execution_kind = "native_agent"
                if execution_kind not in {"native_agent", "sidecar_call"}:
                    continue
                if (
                    execution_kind == "native_agent"
                    and provider_id is not None
                    and provider
                    and not self.provider_agent_capacity_available(
                        provider_id, provider.limits.max_active_agents
                    )
                ):
                    continue
                native_name = "controller-direct" if controller_phase else (
                    registry.native_agent_name(model_id, role)
                    if hasattr(registry, "native_agent_name")
                    else f"brigade-{role}"
                )
                if execution_kind == "sidecar_call" and not controller_phase:
                    native_name = f"sidecar-{role}"
                action_id = (
                    f"action:{run_id}:{epoch_id}:{phase['phase_id']}:{role}:{len(executions)}"
                )
                claim = claims.get(action_id)
                if claim is not None and not include_claimed:
                    continue
                action = {
                    "action_id": action_id,
                    "action_kind": execution_kind,
                    "phase_id": phase["phase_id"],
                    "role": role,
                    "native_agent_name": native_name,
                    "model_id": model_id,
                    "endpoint": route.get("endpoint_override") or "auto",
                    "provider_id": provider_id,
                    "status": "ready",
                    "max_fanout": phase.get("max_fanout"),
                    "current_fanout": len(executions),
                    "requires_main_controller": controller_phase,
                }
                if sidecar_spec is not None:
                    action.update({
                        "sidecar_id": sidecar_id,
                        "sidecar_mode": sidecar_spec.mode,
                        "timeout_seconds": sidecar_spec.timeout_seconds,
                        "max_packet_bytes": sidecar_spec.max_packet_bytes,
                        "max_output_tokens": sidecar_spec.max_output_tokens,
                    })
                if claim is not None:
                    action.update({
                        "status": str(claim["status"]),
                        "claim_token": claim["claim_token"],
                        "reservation_id": claim.get("reservation_id"),
                        "intent_id": claim.get("intent_id"),
                    })
                actions.append(action)
        # One action per role/phase is enough for the controller to ask again;
        # min_fanout is represented by repeated claims, not hidden fanout.
        return actions

    def claim_runnable_action(
        self,
        run_id: str,
        epoch_id: str,
        action_id: str,
        *,
        ttl_seconds: int = 90,
    ) -> dict:
        """Claim one controller-planned native action before spawning it.

        This is deliberately separate from provider admission.  The claim
        prevents duplicate native spawns; the durable provider reservation
        ensures the claim only succeeds when the worker can start now.  A
        denied native Agent call is never queued for later replay.
        """
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        actions = self.get_runnable_actions(run_id, epoch_id, include_claimed=True)
        action = next(
            (item for item in actions if item.get("action_id") == action_id),
            None,
        )
        if action is None:
            raise WorkflowStateError(
                f"runnable action {action_id!r} is no longer available"
            )
        if action.get("status") in {"claimed", "consumed"}:
            return action

        claim_token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        reservation_id = f"action:{action_id}"
        intent_id = f"intent:{action_id}"
        provider_id = action.get("provider_id")
        reservation: dict | None = None
        if provider_id and action.get("action_kind") == "native_agent":
            from enhanced_router.registry import get_registry

            provider = get_registry().providers.get(str(provider_id))
            if provider is None:
                raise WorkflowStateError(f"provider {provider_id!r} is not configured")
            reservation = self.reserve_provider_agent(
                reservation_id=reservation_id,
                run_id=run_id,
                epoch_id=epoch_id,
                provider_id=str(provider_id),
                execution_id=f"pending:{action_id}",
                lane="worker",
                max_active=provider.limits.max_active_agents,
                deadline_at=expires_at,
                reason=f"action-claim:{action_id}",
                enqueue=False,
            )
            if reservation.get("state") != "reserved":
                raise WorkflowStateError(
                    f"provider {provider_id!r} has no capacity for action {action_id!r}"
                )

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            if existing is not None and existing["status"] in {"claimed", "consumed"}:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                return dict(existing)
            pending = conn.execute(
                "SELECT action_id FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND native_agent_name=? AND role=? "
                "AND status IN ('claimed','consumed') AND claude_agent_id IS NULL "
                "AND action_id != ? LIMIT 1",
                (
                    run_id, epoch_id, action["native_agent_name"], action["role"],
                    action_id,
                ),
            ).fetchone()
            if pending is not None:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                raise WorkflowStateError(
                    "a native action with the same name and role is already awaiting "
                    f"lifecycle attachment: {pending[0]}"
                )
            action_kind = str(action.get("action_kind") or "native_agent")
            values = (
                action_id, run_id, epoch_id, action["phase_id"], action["role"],
                action["native_agent_name"], action["model_id"], action_kind, provider_id,
                claim_token, reservation_id if reservation is not None else None,
                intent_id if action_kind == "native_agent" else None,
                "claimed", _utcnow(), _utcnow(), expires_at,
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO runnable_action_claims "
                    "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, action_kind, "
                    "provider_id, claim_token, reservation_id, intent_id, status, created_at, "
                    "claimed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            else:
                conn.execute(
                    "UPDATE runnable_action_claims SET run_id=?, epoch_id=?, phase_id=?, role=?, "
                    "native_agent_name=?, model_id=?, action_kind=?, provider_id=?, claim_token=?, "
                    "reservation_id=?, intent_id=?, status='claimed', claimed_at=?, "
                    "expires_at=?, consumed_at=NULL, claude_agent_id=NULL, execution_id=NULL WHERE action_id=?",
                    (run_id, epoch_id, action["phase_id"], action["role"],
                     action["native_agent_name"], action["model_id"], action_kind, provider_id,
                     claim_token, reservation_id if reservation is not None else None,
                     intent_id if action_kind == "native_agent" else None,
                     values[14], values[15], action_id),
                )
            policy_json = json.dumps(
                {"action_id": action_id, "claim_token": claim_token},
                separators=(",", ":"),
            )
            intent_exists = conn.execute(
                "SELECT 1 FROM spawn_intents WHERE intent_id=?", (intent_id,)
            ).fetchone()
            if action.get("action_kind") != "native_agent":
                pass
            elif intent_exists is None:
                conn.execute(
                    "INSERT INTO spawn_intents "
                    "(intent_id, run_id, epoch_id, phase_id, native_agent_name, role, model_id, "
                    "provider_id, status, created_at, policy_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    "'planned', ?, ?)",
                    (intent_id, run_id, epoch_id, action["phase_id"], action["native_agent_name"],
                     action["role"], action["model_id"], provider_id, _utcnow(), policy_json),
                )
            elif action.get("action_kind") == "native_agent":
                conn.execute(
                    "UPDATE spawn_intents SET status='planned', spawned_at=NULL, completed_at=NULL, "
                    "claude_agent_id=NULL, policy_json=? WHERE intent_id=?",
                    (policy_json, intent_id),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?", (action_id,)
            ).fetchone()
            assert result is not None
            return {**action, **dict(result), "status": "claimed"}
        except Exception:
            conn.rollback()
            if reservation is not None:
                self.release_provider_reservation(reservation_id, "cancelled")
            raise
        finally:
            conn.close()

    def consume_controller_action(
        self, run_id: str, epoch_id: str, action_id: str,
    ) -> dict | None:
        """Consume a claimed controller-integration action exactly once."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? "
                "AND run_id=? AND epoch_id=? AND role='controller' "
                "AND status='claimed' AND expires_at >= ?",
                (action_id, run_id, epoch_id, _utcnow()),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            now = _utcnow()
            conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=? "
                "WHERE action_id=? AND status='claimed'",
                (now, action_id),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

    def finish_controller_action(
        self,
        run_id: str,
        epoch_id: str,
        action_id: str,
        status: str,
    ) -> dict | None:
        """Terminalize a consumed controller-integration action.

        Controller integration is a two-step operation: the controller first
        consumes the claim immediately before acting, then the integration or
        resolution operation records its outcome.  Keeping the claim in
        ``consumed`` after that outcome would make it look active forever and
        would prevent the same candidate from being claimed again after a
        failed integration or an explicit retry decision.
        """
        if status not in {"completed", "failed", "cancelled", "orphaned"}:
            raise ValueError(f"invalid controller action status: {status}")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            updated = conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND run_id=? AND epoch_id=? AND role='controller' "
                "AND action_kind='controller_integration' AND status='consumed'",
                (status, now, action_id, run_id, epoch_id),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return None
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

    def start_sidecar_execution(
        self,
        *,
        run_id: str,
        epoch_id: str,
        action_id: str,
        claim_token: str,
        execution_id: str,
        packet: dict,
    ) -> dict:
        """Atomically consume a sidecar claim and create its execution.

        Sidecars have no Claude Code child process.  They therefore use a
        synthetic agent identity, while retaining the same claim-to-execution
        correlation used by native workers.
        """
        encoded_packet = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        if len(encoded_packet.encode("utf-8")) > 64_000:
            raise WorkflowStateError("sidecar packet exceeds the 64 KiB bound")
        sidecar_agent_id = f"sidecar:{execution_id}"
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? AND run_id=? "
                "AND epoch_id=? AND action_kind='sidecar_call' AND claim_token=? "
                "AND status='claimed' AND expires_at >= ?",
                (action_id, run_id, epoch_id, claim_token, _utcnow()),
            ).fetchone()
            if claim is None:
                raise WorkflowStateError("sidecar claim is missing, expired, or already consumed")
            phase = conn.execute(
                "SELECT status, allowed_roles_json, max_fanout, provider_requirements_json, "
                "max_parallelism, max_attempts "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()
            if phase is None or phase[0] != "active":
                raise WorkflowStateError("sidecar action phase is not active")
            allowed_roles = json.loads(phase[1] or "[]")
            if claim["role"] not in allowed_roles:
                raise WorkflowStateError("sidecar role is not allowed in its phase")
            if claim["provider_id"] not in json.loads(phase[3] or "[]") and json.loads(phase[3] or "[]"):
                raise WorkflowStateError("sidecar provider is not permitted by its phase")
            active_count = conn.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE run_id=? AND epoch_id=? "
                "AND phase_id=? AND status NOT IN ('completed','failed','timeout','cancelled')",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()[0]
            attempt_count = conn.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE run_id=? AND epoch_id=? "
                "AND phase_id=?",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()[0]
            if int(active_count) >= int(phase[4] or phase[2] or 1):
                raise WorkflowStateError("sidecar action exceeds phase parallelism")
            if int(attempt_count) >= int(phase[5] or phase[2] or 1):
                raise WorkflowStateError("sidecar action exceeds phase attempt budget")
            existing = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["epoch_id"] != epoch_id:
                    raise WorkflowStateError("execution ID is owned by another run")
                conn.rollback()
                return dict(existing)
            now = _utcnow()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "status, actor_kind, execution_kind, provider_id, independence_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'started', 'sidecar', 'sidecar_call', ?, ?)",
                (
                    execution_id, run_id, epoch_id, sidecar_agent_id, claim["role"],
                    claim["model_id"], claim["phase_id"], claim["provider_id"],
                    hashlib.sha256(
                        f"sidecar:{claim['model_id']}:{claim['role']}:{claim['phase_id']}".encode()
                    ).hexdigest(),
                ),
            )
            updated = conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                "claude_agent_id=?, execution_id=? WHERE action_id=? AND status='claimed'",
                (now, sidecar_agent_id, execution_id, action_id),
            )
            if updated.rowcount != 1:
                raise WorkflowStateError("sidecar claim changed during execution start")
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, 1, 'started', ?, ?)",
                (execution_id, run_id, epoch_id, encoded_packet, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def start_detached_sidecar_execution(
        self,
        *,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        phase_id: str,
        role: str,
        model_id: str,
        provider_id: str | None,
        packet: dict,
        parent_execution_id: str | None = None,
        retry_count: int = 0,
    ) -> dict:
        """Create a persisted router-owned advisory sidecar execution.

        Fastpath jobs are created by the router itself rather than by a
        workflow claim.  They still use the same execution ledger and event
        stream as claimed sidecars, but are scoped to an existing run/epoch
        and cannot become native-agent or mutation work.
        """
        encoded_packet = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        if len(encoded_packet.encode("utf-8")) > 64_000:
            raise WorkflowStateError("detached sidecar packet exceeds the 64 KiB bound")
        sidecar_agent_id = f"sidecar:{execution_id}"
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            epoch = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if owned is None or epoch is None:
                raise WorkflowStateError("detached sidecar run or epoch is not active")
            if parent_execution_id is not None:
                parent = conn.execute(
                    "SELECT run_id, epoch_id FROM agent_executions WHERE execution_id=?",
                    (parent_execution_id,),
                ).fetchone()
                if parent is None or parent[0] != run_id or parent[1] != epoch_id:
                    raise WorkflowStateError("detached sidecar retry parent is out of scope")
            existing = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["epoch_id"] != epoch_id:
                    raise WorkflowStateError("execution ID is owned by another run")
                conn.rollback()
                return dict(existing)
            now = _utcnow()
            independence_key = hashlib.sha256(
                json.dumps({
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "role": role,
                    "phase_id": phase_id,
                    "execution_id": execution_id,
                }, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "status, actor_kind, execution_kind, provider_id, retry_count, "
                "parent_execution_id, independence_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'started', 'sidecar', 'sidecar_call', ?, ?, ?, ?)",
                (execution_id, run_id, epoch_id, sidecar_agent_id, role, model_id,
                 phase_id, provider_id, retry_count, parent_execution_id, independence_key),
            )
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, 1, 'started', ?, ?)",
                (execution_id, run_id, epoch_id, encoded_packet, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_execution_event(
        self,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        event_type: str,
        payload: dict | None = None,
    ) -> dict:
        """Append an ordered, scoped execution event."""
        payload_json = json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False)
        if len(payload_json.encode("utf-8")) > 64_000:
            raise ValueError("execution event payload exceeds the 64 KiB bound")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM agent_executions WHERE execution_id=? AND run_id=? AND epoch_id=?",
                (execution_id, run_id, epoch_id),
            ).fetchone()
            if owned is None:
                raise WorkflowStateError("execution is not owned by the requested run and epoch")
            seq = int(conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM execution_events WHERE execution_id=?",
                (execution_id,),
            ).fetchone()[0])
            now = _utcnow()
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (execution_id, run_id, epoch_id, seq, event_type, payload_json, now),
            )
            conn.commit()
            return {
                "execution_id": execution_id, "run_id": run_id, "epoch_id": epoch_id,
                "seq": seq, "event_type": event_type, "payload": payload or {},
                "created_at": now,
            }
        finally:
            conn.close()

    def prepare_sidecar_retry(self, execution_id: str) -> dict:
        """Create one bounded retry claim from a failed sidecar execution."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if execution is None:
                raise WorkflowStateError("sidecar execution was not found")
            if execution["execution_kind"] != "sidecar_call":
                raise WorkflowStateError("only sidecar executions can be retried")
            if execution["status"] not in {"failed", "timeout", "cancelled"}:
                raise WorkflowStateError("sidecar execution is not in a retryable terminal state")
            claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE execution_id=? "
                "AND action_kind='sidecar_call' ORDER BY claimed_at DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
            if claim is None:
                raise WorkflowStateError("sidecar execution has no originating claim")
            prior_retries = conn.execute(
                "SELECT COUNT(*) FROM runnable_action_claims WHERE action_id LIKE ?",
                (f"{claim['action_id']}:retry:%",),
            ).fetchone()[0]
            if int(prior_retries) >= 2:
                raise WorkflowStateError("sidecar retry budget exhausted")
            event = conn.execute(
                "SELECT payload_json FROM execution_events WHERE execution_id=? AND seq=1",
                (execution_id,),
            ).fetchone()
            if event is None:
                raise WorkflowStateError("sidecar input packet is unavailable for retry")
            packet = json.loads(str(event[0]))
            retry_action_id = f"{claim['action_id']}:retry:{secrets.token_hex(4)}"
            retry_token = secrets.token_urlsafe(24)
            now = _utcnow()
            expires = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat()
            conn.execute(
                "INSERT INTO runnable_action_claims "
                "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
                "action_kind, provider_id, claim_token, status, created_at, claimed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'sidecar_call', ?, ?, 'claimed', ?, ?, ?)",
                (retry_action_id, execution["run_id"], execution["epoch_id"], execution["phase_id"],
                 execution["role"], claim["native_agent_name"], execution["model_id"],
                 execution["provider_id"], retry_token, now, now, expires),
            )
            conn.commit()
            return {
                "run_id": execution["run_id"], "epoch_id": execution["epoch_id"],
                "action_id": retry_action_id, "claim_token": retry_token, "packet": packet,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_execution_events(
        self,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        *,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Read execution events only within the requested run and epoch."""
        limit = max(1, min(int(limit), 500))
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM execution_events WHERE execution_id=? AND run_id=? "
                "AND epoch_id=? AND seq>? ORDER BY seq LIMIT ?",
                (execution_id, run_id, epoch_id, max(0, int(after_seq)), limit),
            ).fetchall()
            result: list[dict] = []
            for row in rows:
                item = dict(row)
                try:
                    item["payload"] = json.loads(item.pop("payload_json"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    item["payload"] = {"malformed": True}
                result.append(item)
            return result
        finally:
            conn.close()

    def consume_runnable_action_for_spawn(
        self,
        run_id: str,
        epoch_id: str,
        native_agent_name: str,
        *,
        action_id: str | None = None,
        claude_agent_id: str | None = None,
        spawn_call_id: str | None = None,
    ) -> dict | None:
        """Attach a native Agent tool call to a claim before child identity exists.

        PreToolUse only knows the parent tool-call identity.  It records that
        correlation key and leaves the claim pending until SubagentStart can
        attach the actual child.  The explicit ``claude_agent_id`` argument is
        retained for older in-process callers that already have the child ID.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            clauses = [
                "run_id=?", "epoch_id=?", "native_agent_name=?",
                "status='claimed'", "expires_at >= ?",
            ]
            params: list[object] = [run_id, epoch_id, native_agent_name, _utcnow()]
            if action_id:
                clauses.append("action_id=?")
                params.append(action_id)
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE " + " AND ".join(clauses)
                + " ORDER BY claimed_at, action_id LIMIT 1", params,
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            now = _utcnow()
            if claude_agent_id:
                conn.execute(
                    "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                    "claude_agent_id=?, spawn_call_id=COALESCE(?, spawn_call_id) "
                    "WHERE action_id=? AND status='claimed'",
                    (now, claude_agent_id, spawn_call_id, row["action_id"]),
                )
            else:
                conn.execute(
                    "UPDATE runnable_action_claims SET spawn_call_id=? "
                    "WHERE action_id=? AND status='claimed'",
                    (spawn_call_id, row["action_id"]),
                )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status=?, spawned_at=?, claude_agent_id=? "
                    "WHERE intent_id=? AND status='planned'",
                    (
                        "spawned" if claude_agent_id else "planned",
                        now if claude_agent_id else None,
                        claude_agent_id,
                        row["intent_id"],
                    ),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (row["action_id"],),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

    def attach_spawned_agent(
        self,
        *,
        run_id: str,
        epoch_id: str,
        native_agent_name: str,
        role: str,
        model_id: str,
        claude_agent_id: str,
        execution_id: str,
        workspace_id: str | None,
        phase_id: str | None,
        provider_id: str | None = None,
        spawn_call_id: str | None = None,
        claim_token: str | None = None,
        binding_id: int | None = None,
        actor_kind: str = "subagent",
        execution_kind: str = "subagent",
    ) -> dict:
        """Atomically bind a claimed native spawn to its child execution.

        This is the lifecycle seam between PreToolUse and SubagentStart.  It
        refuses ambiguous marker matches, creates the authoritative execution,
        attaches the provider reservation, and only then marks the claim as
        consumed.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            clauses = [
                "run_id=?", "epoch_id=?", "native_agent_name=?", "role=?",
                "status='claimed'", "expires_at >= ?",
            ]
            params: list[object] = [
                run_id, epoch_id, native_agent_name, role, _utcnow(),
            ]
            if claim_token:
                clauses.append("claim_token=?")
                params.append(claim_token)
            if spawn_call_id:
                clauses.append("spawn_call_id=?")
                params.append(spawn_call_id)
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE "
                + " AND ".join(clauses)
                + " ORDER BY claimed_at, action_id",
                params,
            ).fetchall()
            if len(rows) != 1:
                conn.rollback()
                if not rows:
                    raise WorkflowStateError(
                        f"no live claimed native action matches {native_agent_name!r}"
                    )
                raise WorkflowStateError(
                    "native spawn correlation is ambiguous; refusing to choose a claim"
                )
            claim = rows[0]
            if claim["claude_agent_id"] not in (None, claude_agent_id):
                raise WorkflowStateError("native action is already attached to another child")

            existing_execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if existing_execution is not None:
                if (
                    existing_execution["run_id"] != run_id
                    or existing_execution["epoch_id"] != epoch_id
                    or existing_execution["claude_agent_id"] != claude_agent_id
                ):
                    raise WorkflowStateError("execution ID is already owned by another lifecycle")
                conn.rollback()
                return {
                    "claim": dict(claim),
                    "execution": dict(existing_execution),
                }

            if phase_id is not None:
                phase = conn.execute(
                    "SELECT status, allowed_roles_json, required_actor, max_fanout, "
                    "max_parallelism, max_attempts "
                    "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()
                if phase is None or phase[0] != "active":
                    raise WorkflowStateError("native action phase is not active")
                allowed_roles = json.loads(phase[1] or "[]")
                if role not in allowed_roles and not (role == "controller" and phase[2] == "controller"):
                    raise WorkflowStateError("native action role is not allowed in its phase")
                active_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=? "
                    "AND status NOT IN ('completed','failed','timeout','cancelled')",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                attempt_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                if int(active_count) >= int(phase[4] or phase[3] or 1):
                    raise WorkflowStateError("native action exceeds phase parallelism")
                if int(attempt_count) >= int(phase[5] or phase[3] or 1):
                    raise WorkflowStateError("native action exceeds phase attempt budget")

            if binding_id is None:
                existing_binding = self._select_agent_binding(
                    conn, run_id, claude_agent_id,
                )
                if existing_binding is not None:
                    binding_id = int(existing_binding["binding_id"])
            if binding_id is not None:
                binding = conn.execute(
                    "SELECT run_id, epoch_id, claude_agent_id, role, model_id, released_at, "
                    "provider_id FROM agent_bindings WHERE binding_id=?",
                    (binding_id,),
                ).fetchone()
                if binding is None:
                    raise WorkflowStateError(f"unknown agent binding {binding_id}")
                if (
                    binding[0] != run_id or binding[1] != epoch_id
                    or binding[2] != claude_agent_id
                    or binding[3] != role or binding[4] != model_id
                ):
                    raise WorkflowStateError("native execution identity does not match its binding")
                if binding[5] is not None:
                    raise WorkflowStateError("native agent binding is already released")
                if provider_id is not None and binding[6] is not None and provider_id != binding[6]:
                    raise WorkflowStateError("native execution provider does not match its binding")

            independence_key = hashlib.sha256(
                json.dumps({
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "role": role,
                    "phase_id": phase_id,
                    "parent_execution_id": None,
                    "workspace_id": workspace_id,
                }, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

            if workspace_id is not None:
                workspace = conn.execute(
                    "SELECT kind, status, owner_execution_id FROM workspaces WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchone()
                if role in {"implementer", "repairer", "controller"}:
                    if workspace is None or workspace[0] != "shadow" or workspace[1] != "active" \
                            or workspace[2] != execution_id:
                        raise WorkflowStateError(
                            "mutating native execution is not attached to its active shadow workspace"
                        )
            elif role in {"implementer", "repairer", "controller"}:
                raise WorkflowStateError("mutating native execution requires a shadow workspace")

            now = _utcnow()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "binding_id, status, actor_kind, execution_kind, provider_id, workspace_id, independence_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?, ?)",
                (
                    execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                    phase_id, binding_id, actor_kind, execution_kind, provider_id,
                    workspace_id, independence_key,
                ),
            )
            claim_update = conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                "claude_agent_id=?, execution_id=? WHERE action_id=? AND status='claimed'",
                (now, claude_agent_id, execution_id, claim["action_id"]),
            )
            if claim_update.rowcount != 1:
                raise WorkflowStateError("native action claim changed during attachment")
            if claim["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status='spawned', spawned_at=?, claude_agent_id=? "
                    "WHERE intent_id=? AND status IN ('planned','spawned')",
                    (now, claude_agent_id, claim["intent_id"]),
                )
            if claim["reservation_id"]:
                conn.execute(
                    "UPDATE provider_reservations SET execution_id=? "
                    "WHERE reservation_id=? AND execution_id=?",
                    (execution_id, claim["reservation_id"], f"pending:{claim['action_id']}"),
                )
            conn.commit()
            final_claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (claim["action_id"],),
            ).fetchone()
            final_execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            assert final_claim is not None and final_execution is not None
            return {"claim": dict(final_claim), "execution": dict(final_execution)}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_pending_spawn_claim(
        self, run_id: str, epoch_id: str, native_agent_name: str, role: str,
    ) -> dict | None:
        """Return the only unbound claim for a native spawn, if one exists."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND native_agent_name=? AND role=? AND status='claimed' "
                "AND expires_at >= ? ORDER BY claimed_at, action_id",
                (run_id, epoch_id, native_agent_name, role, _utcnow()),
            ).fetchall()
            if len(rows) > 1:
                raise WorkflowStateError(
                    "native spawn correlation is ambiguous; refusing to choose a claim"
                )
            return dict(rows[0]) if rows else None
        finally:
            conn.close()

    def get_unattached_spawn_claim_for_role(
        self, run_id: str, epoch_id: str, role: str,
    ) -> dict | None:
        """Return the pending native claim for a role, if lifecycle is racing."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND role=? AND status='claimed' AND claude_agent_id IS NULL "
                "AND expires_at >= ? ORDER BY claimed_at, action_id",
                (run_id, epoch_id, role, _utcnow()),
            ).fetchall()
            if len(rows) > 1:
                raise WorkflowStateError(
                    "multiple native actions are awaiting attachment for this role"
                )
            return dict(rows[0]) if rows else None
        finally:
            conn.close()

    def fail_spawn_claim(
        self, action_id: str, *, status: str = "failed", reason: str = "",
    ) -> dict | None:
        """Terminally close an unattached spawn claim and its intent."""
        if status not in {"failed", "timed_out", "cancelled", "orphaned"}:
            raise ValueError("invalid unattached spawn claim status")
        conn = self._new_conn()
        reservation_id: str | None = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? "
                "AND status IN ('claimed','consumed')", (action_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            reservation_id = str(row["reservation_id"]) if row["reservation_id"] else None
            now = _utcnow()
            conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND status IN ('claimed','consumed')",
                (status, now, action_id),
            )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status='failed', completed_at=?, policy_json=? "
                    "WHERE intent_id=? AND status IN ('planned','spawned')",
                    (
                        now,
                        json.dumps({"failure": reason}, separators=(",", ":")),
                        row["intent_id"],
                    ),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?", (action_id,)
            ).fetchone()
            result_dict = dict(result) if result else None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if reservation_id:
            reservation_state = {
                "timed_out": "expired",
                "orphaned": "expired",
                "failed": "cancelled",
            }.get(status, status)
            released = self.release_provider_reservation(reservation_id, reservation_state)
            if released:
                from enhanced_router.registry import get_registry
                provider = get_registry().providers.get(str(released.get("provider_id")))
                if provider:
                    self.admit_provider_agents(
                        str(released["provider_id"]), provider.limits.max_active_agents,
                    )
        return result_dict

    def get_spawn_assignment(
        self, run_id: str, epoch_id: str, claude_agent_id: str,
    ) -> dict | None:
        """Return the consumed action identity for a native agent, if any."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND claude_agent_id=? AND status='consumed' "
                "ORDER BY consumed_at DESC LIMIT 1",
                (run_id, epoch_id, claude_agent_id),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def finish_spawn_assignment(
        self,
        run_id: str,
        epoch_id: str,
        claude_agent_id: str,
        status: str,
    ) -> dict | None:
        """Close the action ledger for every native-agent terminal outcome."""
        if status not in {"completed", "failed", "timeout", "cancelled", "error"}:
            raise ValueError(f"invalid native-agent terminal status: {status}")
        claim_status = {
            "completed": "completed",
            "failed": "failed",
            "timeout": "timed_out",
            "cancelled": "cancelled",
            "error": "failed",
        }[status]
        intent_status = "completed" if status == "completed" else "failed"
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND claude_agent_id=? AND status IN ('consumed','claimed') "
                "ORDER BY consumed_at DESC, claimed_at DESC LIMIT 1",
                (run_id, epoch_id, claude_agent_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            now = _utcnow()
            # Older embedded callers may still consume a claim with a child
            # ID before creating an authoritative execution.  Preserve their
            # response compatibility while the production lifecycle (which
            # always sets execution_id through attach_spawned_agent) receives
            # the explicit terminal status.
            terminal_status = "consumed" if (
                claim_status == "completed" and row["execution_id"] is None
            ) else claim_status
            conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND status IN ('consumed','claimed')",
                (terminal_status, now, row["action_id"]),
            )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status=?, completed_at=? WHERE intent_id=? "
                    "AND status IN ('planned','spawned')",
                    (intent_status, now, row["intent_id"]),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (row["action_id"],),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

    def cancel_action_claims(self, run_id: str) -> int:
        """Cancel uncompleted cooperative actions during session teardown."""
        conn = self._new_conn()
        reservation_ids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT intent_id, reservation_id FROM runnable_action_claims "
                "WHERE run_id=? AND status IN ('claimed','consumed')",
                (run_id,),
            ).fetchall()
            reservation_ids = [str(row[1]) for row in rows if row[1]]
            now = _utcnow()
            result = conn.execute(
                "UPDATE runnable_action_claims SET status='cancelled', consumed_at=COALESCE(consumed_at, ?) "
                "WHERE run_id=? AND status IN ('claimed','consumed')",
                (now, run_id),
            )
            for row in rows:
                if row[0]:
                    conn.execute(
                        "UPDATE spawn_intents SET status='cancelled', completed_at=? "
                        "WHERE intent_id=? AND status IN ('planned','spawned')",
                        (now, row[0]),
                    )
            conn.commit()
            count = int(result.rowcount)
        finally:
            conn.close()
        for reservation_id in reservation_ids:
            self.release_provider_reservation(reservation_id, "cancelled")
        return count

    def reconcile_lifecycle(
        self, run_id: str | None = None, *, max_age_seconds: int = 900,
    ) -> dict[str, int]:
        """Reconcile claims, spawn intents, and reservations left by crashes.

        Only records older than ``max_age_seconds`` are considered orphaned so
        a normal lifecycle race is not mistaken for a crash.  The operation is
        idempotent and returns counts suitable for health/status reporting.
        """
        cutoff = _utcnow_age(max_age_seconds)
        conn = self._new_conn()
        reservation_ids: list[str] = []
        provider_ids: set[str] = set()
        orphaned_execution_ids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            run_clause = "" if run_id is None else " AND run_id=?"
            run_params: tuple[object, ...] = () if run_id is None else (run_id,)
            expired = conn.execute(
                "SELECT reservation_id, provider_id FROM provider_reservations "
                "WHERE state IN ('queued','reserved') AND deadline_at IS NOT NULL "
                "AND deadline_at < ?" + run_clause,
                (cutoff, *run_params),
            ).fetchall()
            reservation_ids = [str(row[0]) for row in expired if row[0]]
            provider_ids.update(str(row[1]) for row in expired if row[1])
            claims = conn.execute(
                "SELECT c.action_id, c.reservation_id, c.intent_id, r.provider_id, c.execution_id "
                "FROM runnable_action_claims AS c LEFT JOIN provider_reservations AS r "
                "ON r.reservation_id=c.reservation_id "
                "WHERE c.status IN ('claimed','consumed') AND "
                "COALESCE(c.consumed_at, c.claimed_at) < "
                "?" + (" AND c.run_id=?" if run_id is not None else ""),
                (cutoff, *run_params),
            ).fetchall()
            claim_ids = [str(row[0]) for row in claims]
            claim_reservations = [str(row[1]) for row in claims if row[1]]
            provider_ids.update(str(row[3]) for row in claims if row[3])
            reservation_ids.extend(claim_reservations)
            orphaned_execution_ids = [str(row[4]) for row in claims if row[4]]
            claim_count = 0
            for row in claims:
                action_id, reservation_id, intent_id, _provider_id, _execution_id = row
                conn.execute(
                    "UPDATE runnable_action_claims SET status='orphaned', "
                    "consumed_at=COALESCE(consumed_at, ?) WHERE action_id=? "
                    "AND status IN ('claimed','consumed')",
                    (_utcnow(), action_id),
                )
                if intent_id:
                    conn.execute(
                        "UPDATE spawn_intents SET status='failed', completed_at=? "
                        "WHERE intent_id=? AND status IN ('planned','spawned')",
                        (_utcnow(), intent_id),
                    )
                claim_count += 1
            reservation_count = conn.execute(
                "UPDATE provider_reservations SET state='expired', released_at=? "
                "WHERE state IN ('queued','reserved') AND deadline_at IS NOT NULL "
                "AND deadline_at < ?" + run_clause,
                (_utcnow(), cutoff, *run_params),
            ).rowcount
            conn.commit()
            result = {
                "claims_orphaned": claim_count,
                "reservations_expired": int(reservation_count),
                "spawn_intents_orphaned": len(claim_ids),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if provider_ids:
            from enhanced_router.registry import get_registry
            registry = get_registry()
            for provider_id in provider_ids:
                provider = registry.providers.get(provider_id)
                if provider:
                    self.admit_provider_agents(provider_id, provider.limits.max_active_agents)
        executions_orphaned = 0
        for execution_id in orphaned_execution_ids:
            try:
                self.update_agent_execution(
                    execution_id=execution_id, status="timeout",
                    error="orphaned: no terminal report before crash-recovery cutoff",
                )
                executions_orphaned += 1
            except WorkflowStateError:
                # Already terminal (it finished right before reconciliation
                # ran) -- nothing to reconcile, not a failure.
                pass
        result["executions_orphaned"] = executions_orphaned
        return result

    def admit_provider_agents(self, provider_id: str, max_active: int) -> list[dict]:
        conn = self._new_conn()
        admitted: list[dict] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            active = int(conn.execute(
                "SELECT COUNT(*) FROM provider_reservations WHERE provider_id=? AND state='reserved'",
                (provider_id,),
            ).fetchone()[0])
            capacity = max(0, max_active - active)
            rows = conn.execute(
                "SELECT reservation_id FROM provider_reservations WHERE provider_id=? AND state='queued'"
                " ORDER BY queued_at, reservation_id LIMIT ?", (provider_id, capacity),
            ).fetchall()
            now = _utcnow()
            for row in rows:
                conn.execute(
                    "UPDATE provider_reservations SET state='reserved', admitted_at=? WHERE reservation_id=?",
                    (now, row[0]),
                )
            conn.commit()
            for row in rows:
                item = conn.execute("SELECT * FROM provider_reservations WHERE reservation_id=?", (row[0],)).fetchone()
                if item:
                    admitted.append(dict(item))
            return admitted
        finally:
            conn.close()

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

    def create_spawn_intent(self, **values: object) -> dict:
        required = {"intent_id", "run_id", "epoch_id", "native_agent_name", "role", "status"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"missing spawn intent fields: {sorted(missing)}")
        defaults = {
            "slot": "inherit", "model_id": None, "endpoint_id": None,
            "provider_id": None, "execution_kind": "subagent", "policy_json": "{}",
            "phase_id": None, "fallback_of": None,
        }
        defaults.update(values)
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO spawn_intents (intent_id, run_id, epoch_id, phase_id, native_agent_name,"
                " role, slot, model_id, endpoint_id, provider_id, execution_kind, status, created_at,"
                " fallback_of, policy_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (defaults["intent_id"], defaults["run_id"], defaults["epoch_id"], defaults["phase_id"],
                 defaults["native_agent_name"], defaults["role"], defaults["slot"], defaults["model_id"],
                 defaults["endpoint_id"], defaults["provider_id"], defaults["execution_kind"],
                 defaults["status"], _utcnow(), defaults["fallback_of"], defaults["policy_json"]),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM spawn_intents WHERE intent_id=?", (defaults["intent_id"],)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def update_spawn_intent(self, intent_id: str, status: str, **values: object) -> dict | None:
        if status not in {"planned", "queued", "spawned", "completed", "failed", "cancelled"}:
            raise ValueError("invalid spawn intent status")
        sets = ["status=?"]
        params: list[object] = [status]
        for column in (
            "spawned_at", "completed_at", "fallback_of", "policy_json",
            "claude_agent_id",
        ):
            if column in values:
                sets.append(f"{column}=?")
                params.append(values[column])
        if status == "spawned" and "spawned_at" not in values:
            sets.append("spawned_at=?")
            params.append(_utcnow())
        if status in {"completed", "failed", "cancelled"} and "completed_at" not in values:
            sets.append("completed_at=?")
            params.append(_utcnow())
        params.append(intent_id)
        conn = self._new_conn()
        try:
            conn.execute(f"UPDATE spawn_intents SET {', '.join(sets)} WHERE intent_id=?", params)
            conn.commit()
            row = conn.execute("SELECT * FROM spawn_intents WHERE intent_id=?", (intent_id,)).fetchone()
            return dict(row) if row else None
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

    # ---- Run lifecycle ---------------------------------------------

    def begin_task(
        self,
        run_id: str,
        session_id: str,
        cwd: str,
        workflow_id: str | None = None,
        profile_id: str | None = None,
        signals: list[str] | None = None,
        force_workflow: str | None = None,
        prompt: str = "",
        intake_id: str | None = None,
        minimum_tier: str | None = None,
        baseline_fingerprint: str | None = None,
        contract: dict | None = None,
    ) -> dict:
        """Authoritative task start — atomic run + epoch + phases creation.

        Creates or idempotently confirms the run, creates a fresh epoch
        from the workflow+profile, initializes all workflow phases with
        full semantics, and returns the task contract.

        This is the single authoritative entry point for starting a new
        task/epoch. All hooks call this instead of doing inline
        run/epoch/phase creation.

        Returns dict with: run_id, epoch_id, workflow_id, profile_id,
        phases, specification_hash.
        """
        from enhanced_router.registry import get_registry

        signals = signals or []

        # Determine effective workflow
        if force_workflow:
            effective_workflow = force_workflow
        elif minimum_tier:
            effective_workflow = minimum_tier
        else:
            from enhanced_router.config_models import determine_tier
            tier = determine_tier(signals) if signals else (workflow_id or "normal")
            effective_workflow = tier

        # Verify workflow exists
        reg = get_registry()
        reg.load_workflows()
        spec = reg.get_workflow(effective_workflow)
        if spec is None:
            raise ValueError(f"Workflow '{effective_workflow}' not found in registry")
        reg.load_profiles()
        effective_profile_id = profile_id or spec.default_profile
        profile = reg.get_profile(effective_profile_id)
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # 1. Create or confirm run
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, claude_session_id, cwd, created_at) VALUES (?, ?, ?, ?)",
                (run_id, session_id, cwd, _utcnow()),
            )
            if session_id or cwd:
                sets: list[str] = []
                params: list = []
                if session_id:
                    sets.append("claude_session_id = COALESCE(claude_session_id, ?)")
                    params.append(session_id)
                if cwd:
                    sets.append("cwd = COALESCE(cwd, ?)")
                    params.append(cwd)
                if sets:
                    params.append(run_id)
                    conn.execute(
                        f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ? AND closed_at IS NULL",
                        params,
                    )

            # 2. Check for existing active epoch
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                (run_id,),
            ).fetchone()
            if active:
                conn.rollback()
                raise ValueError(f"Active epoch already exists for run {run_id}")

            # 3. Create epoch
            epoch_id = f"ep_{_utcnow().replace(':', '-').replace('.', '-')}"
            conn.execute(
                "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at,"
                " prompt_digest, minimum_tier, intake_id, contract_json, baseline_fingerprint)"
                " VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, effective_workflow, effective_profile_id, _utcnow(),
                 prompt_digest, minimum_tier or effective_workflow, intake_id,
                 json.dumps(contract or {}, sort_keys=True, separators=(",", ":")),
                 baseline_fingerprint),
            )

            # 4. Set profile routes atomically
            now = _utcnow()
            for role in ("recon", "implementer", "adversary", "repairer"):
                target = profile.route_target(role)
                model_id = target.model
                conn.execute(
                    """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                       VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                       ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                           model_id = excluded.model_id,
                           source = excluded.source,
                           reason = excluded.reason,
                           version = version + 1,
                           changed_at = excluded.changed_at""",
                    (run_id, epoch_id, role, model_id, f"profile:{effective_profile_id}", now),
                )
                conn.execute(
                    "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                )
                conn.execute(
                    "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (json.dumps(target.fallback_models), run_id, epoch_id, role),
                )
                conn.execute(
                    "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                    "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                    (run_id, epoch_id, role, model_id, now),
                )

            # 5. Initialize workflow phases
            phases_data = [
                {
                    "id": p.id,
                    "roles": p.roles,
                    "required": p.required,
                    "mutation": p.mutation,
                    "depends_on": p.depends_on,
                    "conditional": p.conditional,
                    "actor": p.actor or "",
                    "distinct_agent_from": p.distinct_agent_from,
                    "parallel_group": p.parallel_group,
                    "ordinal": p.ordinal,
                    "max_duration_seconds": p.max_duration_seconds,
                    "turn_budget": p.turn_budget,
                    "provider_requirements": p.provider_requirements,
                    "min_fanout": p.min_fanout,
                    "max_fanout": p.max_fanout,
                    "result_schema": p.result_schema,
                    "quality_quorum": p.quality_quorum,
                    "fallback_policy": p.fallback_policy,
                    "execution_kind": p.execution_kind,
                    "max_parallelism": p.max_parallelism,
                    "required_successes": p.required_successes,
                    "max_attempts": p.max_attempts,
                    "max_attempts_per_model": p.max_attempts_per_model,
                    "sidecar_id": p.sidecar,
                }
                for p in spec.phases
            ]

            for phase in phases_data:
                canonical = json.dumps({k: v for k, v in sorted(phase.items())}, sort_keys=True)
                spec_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                conn.execute(
                    """INSERT INTO workflow_phases
                       (run_id, epoch_id, phase_id, status, actor, required, mutating,
                       allowed_roles_json, dependencies_json, condition_json, parallel_group,
                       specification_hash, ordinal, distinct_agent_from_json, max_duration_seconds,
                       turn_budget, provider_requirements_json, min_fanout, max_fanout, result_schema,
                       quality_quorum, fallback_policy, execution_kind, required_actor,
                       max_parallelism, required_successes, max_attempts, max_attempts_per_model, sidecar_id)
                       VALUES (
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?, ?
                       )""",
                    (
                        run_id, epoch_id, phase["id"], "pending",
                        phase.get("actor", ""),
                        1 if phase.get("required", True) else 0,
                        1 if phase.get("mutation", False) else 0,
                        json.dumps(phase.get("roles", [])),
                        json.dumps(phase.get("depends_on", [])),
                        json.dumps(phase.get("conditional")) if phase.get("conditional") else None,
                        phase.get("parallel_group"),
                        spec_hash,
                        phase.get("ordinal"), json.dumps(phase.get("distinct_agent_from", [])),
                        phase.get("max_duration_seconds"), phase.get("turn_budget"),
                        json.dumps(phase.get("provider_requirements", [])), phase.get("min_fanout", 1),
                        phase.get("max_fanout", 1), phase.get("result_schema"),
                        phase.get("quality_quorum", 1), phase.get("fallback_policy"),
                        phase.get("execution_kind", "native_agent"), phase.get("actor", ""),
                        phase.get("max_parallelism"), phase.get("required_successes"),
                        phase.get("max_attempts") or phase.get("max_fanout", 1), phase.get("max_attempts_per_model"),
                        phase.get("sidecar_id"),
                    ),
                )

            conn.commit()

            # 6. Controller policy is capability-based. The controller is not
            # required to be one of the worker models in the selected profile.
            from enhanced_router.registry import get_registry
            registry = get_registry()
            permitted_models = [model_id for model_id, _ in registry.controller_models()]
            permitted_models.extend(
                model_id for model_id, model in registry.models.items()
                if model.enabled and model.backend == "anthropic-passthrough"
            )
            if not permitted_models:
                # Backward-compatible fallback for catalogs not yet certified.
                permitted_models = sorted({profile.route_target(role).model for role in _VALID_ROLES})
            self.upsert_controller_policy(run_id, permitted_models, policy="reject")

            # Return the task contract
            phases = self.get_workflow_phases(run_id, epoch_id)
            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "workflow_id": effective_workflow,
                "profile_id": effective_profile_id,
                "intake_id": intake_id,
                "minimum_tier": minimum_tier or effective_workflow,
                "prompt_digest": prompt_digest,
                "phases": [
                    {
                        "phase_id": p["phase_id"],
                        "status": p["status"],
                        "required": bool(p["required"]),
                        "mutating": bool(p["mutating"]),
                        "allowed_roles": __import__("json").loads(p["allowed_roles_json"]),
                        "depends_on": __import__("json").loads(p["dependencies_json"]),
                        "specification_hash": p["specification_hash"],
                    }
                    for p in phases
                ],
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def validate_completion(
        self,
        run_id: str,
        epoch_id: str,
        parsed: dict[str, str],
        session_dir: 'Path | None' = None,
    ) -> dict:
        """Validate completion evidence against authoritative state.

        Checks: tier/workflow consistency, phase sequence, findings
        lifecycle, workspace fingerprint, and route snapshot.

        Returns dict with: valid (bool), reason (str|None),
        snapshot_sha256 (str|None).
        """

        # 1. Validate workflow tier matches active epoch
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            return {"valid": False, "reason": f"No active epoch for run {run_id}"}

        # Lifecycle state is authoritative.  JSONL hook mirrors and the
        # assistant's footer cannot make an active or queued execution look
        # complete.  The controller's own provider reservation is retained
        # for the lifetime of its session, so only worker reservations are
        # considered here; they must have been released by SubagentStop.
        active_executions = self.get_agent_executions(run_id, epoch_id=epoch_id)
        active_executions = [
            execution for execution in active_executions
            if execution.get("status") not in {"completed", "failed", "timeout", "cancelled"}
        ]
        if active_executions:
            return {
                "valid": False,
                "reason": "Active native agent execution(s) remain: "
                + ", ".join(str(item.get("execution_id")) for item in active_executions),
            }
        worker_reservations = [
            reservation for reservation in self.get_provider_reservations(active_only=True)
            if reservation.get("run_id") == run_id
            and reservation.get("epoch_id") == epoch_id
            and reservation.get("lane") != "controller"
        ]
        if worker_reservations:
            return {
                "valid": False,
                "reason": "Provider agent reservations remain active or queued: "
                + ", ".join(str(item.get("reservation_id")) for item in worker_reservations),
            }
        if self.get_active_mutation_leases(run_id, epoch_id):
            return {"valid": False, "reason": "An active workspace mutation lease remains"}
        workspace_rows = self.get_workspaces(run_id=run_id, epoch_id=epoch_id)
        pending_workspaces = [
            row for row in workspace_rows
            if row.get("kind") == "shadow" and row.get("status") in {"active", "ready"}
        ]
        if pending_workspaces:
            return {
                "valid": False,
                "reason": "Shadow workspace changesets still require integration: "
                + ", ".join(str(row.get("workspace_id")) for row in pending_workspaces),
            }
        conn = self._new_conn()
        try:
            red_candidates = conn.execute(
                "SELECT candidate_id, disposition FROM integration_candidates "
                "WHERE run_id=? AND epoch_id=? AND disposition IN ('red','yellow','pending')",
                (run_id, epoch_id),
            ).fetchall()
        finally:
            conn.close()
        if red_candidates:
            return {
                "valid": False,
                "reason": "Integration candidates require main-controller adjudication: "
                + ", ".join(f"{row[0]} ({row[1]})" for row in red_candidates),
            }
        reported_workflow = parsed.get("Workflow-ID", "")
        if reported_workflow and reported_workflow != active.get("workflow_id"):
            return {
                "valid": False,
                "reason": (
                    f"Workflow-ID mismatch: message says '{reported_workflow}' "
                    f"but active epoch has workflow '{active['workflow_id']}'"
                ),
            }

        reported_tier = parsed.get("Workflow-Tier", "")
        if reported_tier and reported_tier != active.get("workflow_id"):
            return {
                "valid": False,
                "reason": (
                    f"Workflow-Tier mismatch: message says '{reported_tier}' "
                    f"but active epoch has workflow '{active.get('workflow_id')}'"
                ),
            }

        # 2. Validate all required phases completed, no active phases remain
        phases = self.get_workflow_phases(run_id, epoch_id)
        for phase in phases:
            if phase.get("required") and phase.get("status") != "completed":
                return {
                    "valid": False,
                    "reason": f"Required phase '{phase['phase_id']}' is not completed (status: {phase.get('status')})"
                }
            if phase.get("status") == "active":
                return {
                    "valid": False,
                    "reason": f"Phase '{phase['phase_id']}' is still active at completion"
                }

        # 3. Validate phase sequence: dependencies satisfied
        for phase in phases:
            if phase.get("status") == "completed":
                deps = json.loads(phase.get("dependencies_json", "[]"))
                for dep_id in deps:
                    dep_phase = next((p for p in phases if p["phase_id"] == dep_id), None)
                    if dep_phase is None or dep_phase.get("status") not in {"completed", "skipped"}:
                        return {
                            "valid": False,
                            "reason": f"Phase '{phase['phase_id']}' completed but dependency '{dep_id}' is not completed"
                        }

        # 4. Validate mutating phases don't overlap
        # mutating_phases = [p for p in phases if p.get("mutating") and p.get("status") == "completed"]
        # This is a simple check - more thorough overlap detection would require timestamps
        # For now, ensure no two mutating phases have overlapping active periods by checking
        # that they completed sequentially (simplified)

        # 5. Validate findings lifecycle via state
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        open_accepted = [f for f in findings if f.get("disposition") == "accepted" and f.get("verification_status") == "pending"]

        accepted_findings = [f for f in findings if f.get("disposition") == "accepted"]
        if open_accepted:
            return {
                "valid": False,
                "reason": f"Accepted findings remain unresolved: {len(open_accepted)} open accepted finding(s)",
            }
        # The footer is presentation only. The database is authoritative: an
        # empty accepted set is valid, and a non-empty set must be verified.
        if any(f.get("verification_status") != "verified" for f in accepted_findings):
            return {"valid": False, "reason": "Accepted findings exist without verified resolution evidence"}

        # 6. Validate that phases have result_evidence when completed
        for phase in phases:
            if phase.get("status") == "completed" and not phase.get("result_evidence"):
                return {
                    "valid": False,
                    "reason": f"Phase '{phase['phase_id']}' is completed but has no result_evidence"
                }

        # 7. Validate route snapshot
        snapshot_sha256 = parsed.get("Route-Snapshot-SHA256", "").lower()
        if not snapshot_sha256:
            return {"valid": False, "reason": "Route-Snapshot-SHA256 is required"}
        actual_snapshot = self.create_route_snapshot(run_id, epoch_id, purpose="verified-completion")
        if snapshot_sha256 != actual_snapshot:
            return {
                "valid": False,
                "reason": f"Route snapshot mismatch: reported {snapshot_sha256} but current state produces {actual_snapshot}",
                "snapshot_sha256": actual_snapshot,
            }

        return {"valid": True, "reason": None, "snapshot_sha256": snapshot_sha256 or None}

    def prepare_completion_token(
        self,
        run_id: str,
        epoch_id: str,
        workspace_fingerprint: str,
        *,
        ttl_seconds: int = 300,
    ) -> dict:
        """Issue a short-lived, state-generated completion attestation."""
        if not hmac.compare_digest(
            workspace_fingerprint.lower(), workspace_fingerprint
        ) or len(workspace_fingerprint) != 64:
            raise WorkflowStateError("workspace fingerprint must be 64 lowercase hexadecimal characters")
        try:
            int(workspace_fingerprint, 16)
        except ValueError as exc:
            raise WorkflowStateError("workspace fingerprint must be hexadecimal") from exc
        if ttl_seconds < 1:
            raise ValueError("completion token TTL must be positive")
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            raise WorkflowStateError("completion token requires the active epoch")
        snapshot = self.create_route_snapshot(run_id, epoch_id, purpose="verified-completion")
        token = secrets.token_urlsafe(32)
        token_id = f"completion:{secrets.token_hex(12)}"
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=ttl_seconds)
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO completion_tokens "
                "(token_id, run_id, epoch_id, token_hash, workspace_fingerprint, "
                "route_snapshot_sha256, issued_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    token_id, run_id, epoch_id,
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    workspace_fingerprint, snapshot,
                    issued_at.isoformat(), expires_at.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "token_id": token_id,
            "token": token,
            "run_id": run_id,
            "epoch_id": epoch_id,
            "workspace_fingerprint": workspace_fingerprint,
            "route_snapshot_sha256": snapshot,
            "expires_at": expires_at.isoformat(),
        }

    def consume_completion_token(
        self,
        run_id: str,
        epoch_id: str,
        token: str,
        workspace_fingerprint: str,
        route_snapshot_sha256: str,
    ) -> dict:
        """Consume a completion attestation exactly once and verify its binding."""
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM completion_tokens WHERE run_id=? AND epoch_id=? "
                "AND token_hash=?",
                (run_id, epoch_id, token_hash),
            ).fetchone()
            if row is None:
                return {"valid": False, "reason": "completion token not found"}
            if row["consumed_at"] is not None:
                return {"valid": False, "reason": "completion token was already consumed"}
            if datetime.fromisoformat(str(row["expires_at"])) < datetime.now(timezone.utc):
                return {"valid": False, "reason": "completion token expired"}
            if not hmac.compare_digest(str(row["workspace_fingerprint"]), workspace_fingerprint):
                return {"valid": False, "reason": "completion token workspace mismatch"}
            if not hmac.compare_digest(str(row["route_snapshot_sha256"]), route_snapshot_sha256):
                return {"valid": False, "reason": "completion token route snapshot mismatch"}
            consumed_at = _utcnow()
            updated = conn.execute(
                "UPDATE completion_tokens SET consumed_at=? "
                "WHERE token_id=? AND consumed_at IS NULL",
                (consumed_at, row["token_id"]),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return {"valid": False, "reason": "completion token was already consumed"}
            conn.commit()
            return {"valid": True, "token_id": row["token_id"], "consumed_at": consumed_at}
        finally:
            conn.close()

    def create_run(
        self,
        run_id: str,
        session_id: str | None = None,
        cwd: str | None = None,
        controller_capability: str | None = None,
    ) -> dict:
        """Insert run if not exists (idempotent). Returns run dict."""
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, claude_session_id, cwd, controller_capability_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    session_id,
                    cwd,
                    hashlib.sha256(controller_capability.encode("utf-8")).hexdigest()
                    if controller_capability else None,
                    _utcnow(),
                ),
            )
            if controller_capability:
                conn.execute(
                    "UPDATE runs SET controller_capability_hash=COALESCE(controller_capability_hash, ?) "
                    "WHERE run_id=?",
                    (
                        hashlib.sha256(controller_capability.encode("utf-8")).hexdigest(),
                        run_id,
                    ),
                )
            if session_id is not None or cwd is not None:
                conn.execute(
                    "UPDATE runs SET claude_session_id=COALESCE(claude_session_id, ?), "
                    "cwd=COALESCE(cwd, ?) WHERE run_id=?",
                    (session_id, cwd, run_id),
                )
            conn.commit()
            row = conn.execute(
                "SELECT run_id, claude_session_id, cwd, controller_capability_hash, "
                "created_at, closed_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to create run {run_id}")
            return dict(zip(
                (
                    "run_id", "claude_session_id", "cwd",
                    "controller_capability_hash", "created_at", "closed_at",
                ),
                row,
            ))
        finally:
            conn.close()

    def get_run(self, run_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, claude_session_id, cwd, controller_capability_hash, "
                "created_at, closed_at FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                (
                    "run_id", "claude_session_id", "cwd",
                    "controller_capability_hash", "created_at", "closed_at",
                ),
                row,
            ))
        finally:
            conn.close()

    def verify_controller_capability(
        self, run_id: str, capability: str, *, session_id: str | None = None,
    ) -> bool:
        """Validate the ephemeral controller credential for one run."""
        run = self.get_run(run_id)
        if run is None or run.get("closed_at"):
            return False
        if session_id and run.get("claude_session_id") != session_id:
            return False
        expected = str(run.get("controller_capability_hash") or "")
        actual = hashlib.sha256(capability.encode("utf-8")).hexdigest()
        return bool(expected) and hmac.compare_digest(expected, actual)

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
                "SELECT * "
                "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
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
        """Set status='closed' and closed_at. Release orphaned bindings."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE epochs SET status = 'closed', closed_at = ? WHERE run_id = ? AND epoch_id = ?",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE agent_bindings SET released_at = ? WHERE run_id = ? AND epoch_id = ? AND released_at IS NULL",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE controller_bindings SET released_at=? WHERE run_id=? AND released_at IS NULL",
                (_utcnow(), run_id),
            )
            conn.execute(
                "UPDATE provider_reservations SET state='released', released_at=? "
                "WHERE run_id=? AND epoch_id=? AND state IN ('queued','reserved')",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE mutation_leases SET released_at=? WHERE run_id=? AND epoch_id=? AND released_at IS NULL",
                (_utcnow(), run_id, epoch_id),
            )
            conn.execute(
                "UPDATE agent_executions SET status='cancelled', completed_at=?, updated_at=? "
                "WHERE run_id=? AND epoch_id=? AND status IN ('started','running')",
                (_utcnow(), _utcnow(), run_id, epoch_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_profile_routes_atomic(
        self, run_id: str, epoch_id: str, profile_id: str, reason: str
    ) -> dict[str, dict]:
        """BEGIN IMMEDIATE transaction: set all 4 role routes from profile.

        Updates epochs.profile_id, upserts each role route, appends a
        *route_event* row per changed role, and returns the actual persisted
        versions.  One failure rolls back ALL changes.
        """

        from enhanced_router.registry import get_registry

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                now = _utcnow()
                reg = get_registry()
                reg.load_profiles()
                profile = reg.get_profile(profile_id)

                # 1. Update epochs.profile_id on the active epoch row
                conn.execute(
                    "UPDATE epochs SET profile_id = ? WHERE run_id = ? AND epoch_id = ? AND closed_at IS NULL",
                    (profile_id, run_id, epoch_id),
                )

                results: dict[str, dict] = {}
                for role in ("recon", "implementer", "adversary", "repairer"):
                    target = profile.route_target(role)
                    model_id = target.model
                    conn.execute(
                        """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                           VALUES (?, ?, ?, ?, 'profile', ?, 1, ?)
                           ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                               model_id = excluded.model_id,
                               source = excluded.source,
                               reason = excluded.reason,
                               version = version + 1,
                               changed_at = excluded.changed_at""",
                        (run_id, epoch_id, role, model_id, reason, now),
                    )
                    conn.execute(
                        "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                    )
                    conn.execute(
                        "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (json.dumps(target.fallback_models), run_id, epoch_id, role),
                    )

                    # 2. Read actual version after upsert
                    row = conn.execute(
                        "SELECT version, model_id FROM role_routes WHERE run_id=? AND epoch_id=? AND role=?",
                        (run_id, epoch_id, role),
                    ).fetchone()
                    actual_version = row[0] if row else 1

                    # 3. Append route_event for this role
                    conn.execute(
                        "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                        "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                        (run_id, epoch_id, role, model_id, now),
                    )

                    results[role] = {
                        "run_id": run_id,
                        "epoch_id": epoch_id,
                        "role": role,
                        "model_id": model_id,
                        "source": "profile",
                        "reason": reason,
                        "version": actual_version,
                    }

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return results
        finally:
            conn.close()

    def create_epoch_from_profile(
        self, run_id: str, epoch_id: str, workflow_id: str, profile_id: str
    ) -> dict:
        """Transaction: create_epoch + set profile routes. Returns epoch dict.

        Routes are NOT optional -- a profile load failure propagates as an
        exception so the caller knows the epoch has no routes.
        """
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
                    raise ValueError(
                        f"Active epoch already exists for run {run_id}"
                    )

                # Create the epoch row
                conn.execute(
                    "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at) VALUES (?, ?, ?, ?, 'active', ?)",
                    (run_id, epoch_id, workflow_id, profile_id, _utcnow()),
                )

                # Load profile and set routes -- raise on failure
                from enhanced_router.registry import get_registry

                reg = get_registry()
                reg.load_profiles()
                profile = reg.get_profile(profile_id)
                now = _utcnow()
                for role in ("recon", "implementer", "adversary", "repairer"):
                    target = profile.route_target(role)
                    model_id = target.model
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
                    conn.execute(
                        "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                    )
                    conn.execute(
                        "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                        (json.dumps(target.fallback_models), run_id, epoch_id, role),
                    )

                    # Append route_event for each role
                    conn.execute(
                        "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                        "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                        (run_id, epoch_id, role, model_id, now),
                    )

                conn.commit()
                epoch = conn.execute(
                    "SELECT id, run_id, epoch_id, workflow_id, profile_id, status, created_at, closed_at "
                    "FROM epochs WHERE run_id = ? AND closed_at IS NULL LIMIT 1",
                    (run_id,),
                ).fetchone()
                if epoch is None:
                    raise RuntimeError(
                        f"Failed to create epoch {epoch_id} for run {run_id}"
                    )
                return dict(
                    zip(
                        (
                            "id",
                            "run_id",
                            "epoch_id",
                            "workflow_id",
                            "profile_id",
                            "status",
                            "created_at",
                            "closed_at",
                        ),
                        epoch,
                    )
                )
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    # ---- Route operations ------------------------------------------

    def set_role_route(
        self, run_id: str, epoch_id: str, role: str, model_id: str, source: str,
        reason: str = "", endpoint_override: str | None = None,
        fallback_models: list[str] | None = None,
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
            if fallback_models is not None:
                conn.execute(
                    "UPDATE role_routes SET fallback_models_json=? "
                    "WHERE run_id=? AND epoch_id=? AND role=?",
                    (json.dumps(fallback_models), run_id, epoch_id, role),
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
                "endpoint_override": endpoint_override,
                "fallback_models": fallback_models or [],
            }
        finally:
            conn.close()

    def get_role_route(self, run_id: str, epoch_id: str, role: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at, endpoint_id, endpoint_override, fallback_models_json "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at", "endpoint_id", "endpoint_override", "fallback_models_json"), row
            ))
        finally:
            conn.close()

    def get_epoch_routes(self, run_id: str, epoch_id: str) -> dict[str, dict]:
        """Return dict of {role: {model_id, version, ...}}."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at, endpoint_id, endpoint_override, fallback_models_json "
                "FROM role_routes WHERE run_id = ? AND epoch_id = ? ORDER BY role",
                (run_id, epoch_id),
            ).fetchall()
            result: dict[str, dict] = {}
            for row in rows:
                role = row[2]
                result[role] = dict(zip(
                    ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at", "endpoint_id", "endpoint_override", "fallback_models_json"), row
                ))
            return result
        finally:
            conn.close()

    # ---- Binding operations ----------------------------------------

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

    # ---- Controller bindings and endpoint observations ----------------

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
                " allowed_deployments_json, deployment_policy_digest, bound_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, client_session_id, public_model, registry_model_id, backend, upstream_model,
                 provider_id, api_base, catalog_generation, registry_hash, certification_id, auth_spec_json,
                 api_key_env, endpoint_id, provider_ids_json, endpoint_selection_reason, endpoint_policy_json,
                litellm_model_name, configuration_hash, routing_mode, deployment_group,
                allowed_deployments_json, deployment_policy_digest, now),
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

    # ---- Snapshot --------------------------------------------------

    def create_route_snapshot(
        self, run_id: str, epoch_id: str, purpose: str = "completion"
    ) -> str:
        """Capture state as JSON, SHA-256. Returns hex digest.

        Deterministic: no timestamp in hash input, ALL bindings included
        (not just active), sorted keys, compact separators.
        """
        conn = self._new_conn()
        try:
            routes_rows = conn.execute(
                "SELECT role, model_id, version, endpoint_id, fallback_models_json FROM role_routes "
                "WHERE run_id = ? AND epoch_id = ?",
                (run_id, epoch_id),
            ).fetchall()
            routes = {
                row[0]: {
                    "model_id": row[1],
                    "version": row[2],
                    "endpoint_id": row[3],
                    "fallback_models": json.loads(row[4] or "[]"),
                }
                for row in routes_rows
            }

            # ALL bindings, not just active ones
            bindings_rows = conn.execute(
                "SELECT claude_agent_id, role, model_id, binding_id, released_at, backend, "
                "endpoint_id, upstream_model, api_base, catalog_generation, registry_hash, "
                "certification_id, provider_id, endpoint_selection_reason, configuration_hash "
                "FROM agent_bindings WHERE run_id = ? AND epoch_id = ?",
                (run_id, epoch_id),
            ).fetchall()
            bindings = [
                {
                    "claude_agent_id": row[0],
                    "role": row[1],
                    "model_id": row[2],
                    "binding_id": row[3],
                    "released_at": row[4],
                    "backend": row[5],
                    "endpoint_id": row[6],
                    "upstream_model": row[7],
                    "api_base": row[8],
                    "catalog_generation": row[9],
                    "registry_hash": row[10],
                    "certification_id": row[11],
                    "provider_id": row[12],
                    "endpoint_selection_reason": row[13],
                    "configuration_hash": row[14],
                }
                for row in bindings_rows
            ]

            max_event = conn.execute(
                "SELECT MAX(id) FROM route_events WHERE run_id = ? AND epoch_id = ?",
                (run_id, epoch_id),
            ).fetchone()[0]  # type: ignore[index]

            snapshot_data = {
                "purpose": purpose,
                "run_id": run_id,
                "epoch_id": epoch_id,
                "routes": routes,
                "all_bindings": bindings,
                "max_event_id": max_event,
            }

            json_bytes = json.dumps(
                snapshot_data, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
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

    # ---- LiteLLM catalog lifecycle -----------------------------------

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

    def count_active_bindings_for_generation(self, generation: int) -> int:
        """Return the number of unreleased agent bindings pinned to a generation."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM agent_bindings "
                "WHERE catalog_generation = ? AND released_at IS NULL",
                (generation,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def fail_litellm_generation_executions(self, generation: int) -> int:
        """Fail active executions pinned to a generation that was force-drained."""
        conn = self._new_conn()
        try:
            now = _utcnow()
            cursor = conn.execute(
                "UPDATE agent_executions SET status='failed', completed_at=?, "
                "updated_at=?, error_class='generation_drain_timeout', "
                "error='LiteLLM generation was force-drained while active' "
                "WHERE status IN ('started','running') AND binding_id IN ("
                "SELECT binding_id FROM agent_bindings WHERE catalog_generation=?"
                ")",
                (now, now, generation),
            )
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()

    # ---- CAS route updates -------------------------------------------

    def set_role_route_cas(
        self,
        run_id: str,
        epoch_id: str,
        role: str,
        model_id: str,
        command_id: str,
        expected_version: int,
        actor_type: str = "controller",
        reason: str = "",
    ) -> dict:
        """Compare-and-set route update with idempotency on command_id.

        Raises ``RouteConflictError`` when the current version does not
        match *expected_version* or the epoch is closed.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {_VALID_ROLES}")

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()

            # Verify epoch is active
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND epoch_id = ? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if not active:
                conn.rollback()
                raise ValueError("No active epoch")

            # Check idempotency: has this command_id already been applied?
            idem = conn.execute(
                "SELECT id FROM binding_commands "
                "WHERE command_id = ? AND status = 'applied' "
                "AND command_type = 'route_change'",
                (command_id,),
            ).fetchone()
            if idem is not None:
                # Return existing result
                row = conn.execute(
                    "SELECT run_id, epoch_id, role, model_id, source, reason, version, changed_at "
                    "FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                    (run_id, epoch_id, role),
                ).fetchone()
                return dict(zip(
                    ("run_id", "epoch_id", "role", "model_id", "source", "reason", "version", "changed_at"),
                    row,
                )) | {"idempotent": True}

            # Fetch current row so we know old model_id and version
            existing = conn.execute(
                "SELECT version, model_id FROM role_routes WHERE run_id = ? AND epoch_id = ? AND role = ?",
                (run_id, epoch_id, role),
            ).fetchone()
            if existing is None:
                conn.rollback()
                raise ValueError(f"No existing route for {run_id}/{epoch_id}/{role}")

            current_version = existing[0]
            if current_version != expected_version:
                conn.rollback()
                raise RouteConflictError(
                    f"CAS conflict: expected version {expected_version}, "
                    f"but current version is {current_version}"
                )

            new_version = current_version + 1
            old_model_id = existing[1]

            conn.execute(
                """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                       model_id = excluded.model_id,
                       source = excluded.source,
                       reason = excluded.reason,
                       version = excluded.version,
                       changed_at = excluded.changed_at""",
                (run_id, epoch_id, role, model_id, "cas", reason, new_version, now),
            )

            conn.execute(
                "INSERT INTO route_events (run_id, epoch_id, event_type, role, old_model_id, new_model_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, "route_change", role, old_model_id, model_id, now),
            )

            # Mark command as applied
            conn.execute(
                "INSERT INTO binding_commands (command_id, run_id, epoch_id, command_type, actor_type, reason, status, applied_at, created_at) "
                "VALUES (?, ?, ?, 'route_change', ?, ?, 'applied', ?, ?)",
                (command_id, run_id, epoch_id, actor_type, reason, now, now),
            )

            conn.commit()
            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "role": role,
                "model_id": model_id,
                "source": "cas",
                "reason": reason,
                "version": new_version,
                "changed_at": now,
                "idempotent": False,
            }
        finally:
            conn.close()

    # ---- Controller policy -------------------------------------------

    def upsert_controller_policy(
        self,
        run_id: str,
        model_ids: list[str],
        policy: str = "reject",
    ) -> None:
        """Store or update a controller model policy."""
        if policy not in ("allow", "reject"):
            raise ValueError(f"Invalid policy: {policy}. Must be 'allow' or 'reject'")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            conn.execute(
                """INSERT INTO controller_policies (run_id, permitted_models, model_change_policy, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       permitted_models = excluded.permitted_models,
                       model_change_policy = excluded.model_change_policy,
                       updated_at = excluded.updated_at""",
                (run_id, json.dumps(model_ids), policy, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_controller_policy(self, run_id: str) -> dict | None:
        """Return the controller policy for *run_id*, or None.

        for the public API.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, permitted_models, model_change_policy, created_at, updated_at "
                "FROM controller_policies WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "permitted_models", "model_change_policy", "created_at", "updated_at"), row
            ))
        finally:
            conn.close()

    def validate_controller_model(self, run_id: str, model_id: str) -> bool:
        """Check whether *model_id* is permitted under the controller policy.

        Returns ``True`` when the model is allowed (or no policy is set).
        Raises ``ControllerModelError`` when the model is rejected.
        """
        try:
            from enhanced_router.registry import get_registry
            spec = get_registry().get_model(model_id)
            if spec.enabled and (
                spec.capabilities.controller_eligible
                or spec.backend == "anthropic-passthrough"
            ):
                return True
        except Exception:
            pass
        policy = self.get_controller_policy(run_id)
        if policy is None:
            return True  # no policy means allow
        if policy["model_change_policy"] == "allow":
            return True
        permitted = json.loads(policy["permitted_models"])
        if model_id in permitted:
            return True
        raise ControllerModelError(f"Model '{model_id}' is not permitted by controller policy")

    # ---- Mutation leases ---------------------------------------------

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
        """
        lease_workspace = workspace_id or run_id
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()

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

    # ---- Shadow workspace / changeset lifecycle ------------------------

    def create_workspace(
        self,
        *,
        workspace_id: str,
        run_id: str,
        epoch_id: str,
        kind: str,
        path: str,
        base_sha: str | None,
        dirty_patch_hash: str | None,
        status: str = "active",
        owner_execution_id: str | None = None,
        baseline_untracked_files: list[str] | None = None,
        parent_canonical_generation: int | None = None,
        parent_dirty_patch_hash: str | None = None,
    ) -> dict:
        """Register a main, shadow, or integration Git workspace."""
        if kind not in {"main", "shadow", "integration"}:
            raise ValueError(f"invalid workspace kind: {kind}")
        if status not in {"active", "ready", "merged", "discarded", "failed"}:
            raise ValueError(f"invalid workspace status: {status}")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    parent_canonical_generation, parent_dirty_patch_hash,
                    status, owner_execution_id, baseline_untracked_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (workspace_id, run_id, epoch_id, kind, path, base_sha,
                 dirty_patch_hash, base_sha, dirty_patch_hash,
                 parent_canonical_generation, parent_dirty_patch_hash,
                 status, owner_execution_id,
                 json.dumps(sorted(baseline_untracked_files or [])), _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register_main_workspace(
        self,
        *,
        workspace_id: str,
        run_id: str,
        epoch_id: str,
        path: str,
        base_sha: str,
        dirty_patch_hash: str,
        baseline_untracked_files: list[str] | None = None,
    ) -> dict:
        """Register one canonical checkout and reject cross-run ownership."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=? AND kind='main' "
                "AND status IN ('active','ready') LIMIT 1",
                (workspace_id,),
            ).fetchone()
            if row is not None:
                existing = dict(row)
                if existing["run_id"] != run_id:
                    raise WorkflowStateError(
                        f"canonical workspace is already active for run {existing['run_id']}"
                    )
                current_base_sha = existing.get("current_base_sha") or existing.get("base_sha")
                current_dirty_hash = existing.get("current_dirty_hash") or existing.get("dirty_patch_hash")
                if current_base_sha != base_sha or current_dirty_hash != dirty_patch_hash:
                    raise WorkflowStateError("canonical workspace baseline changed during the run")
                if epoch_id and existing.get("epoch_id") != epoch_id and existing.get("epoch_id") == "session-intake":
                    conn.execute(
                        "UPDATE workspaces SET epoch_id=? WHERE workspace_id=?",
                        (epoch_id, workspace_id),
                    )
                    existing["epoch_id"] = epoch_id
                conn.commit()
                return existing
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    canonical_generation, status, baseline_untracked_json, created_at)
                   VALUES (?, ?, ?, 'main', ?, ?, ?, ?, ?, 0, 'active', ?, ?)""",
                (workspace_id, run_id, epoch_id, path, base_sha, dirty_patch_hash,
                 base_sha, dirty_patch_hash,
                 json.dumps(sorted(baseline_untracked_files or [])), _utcnow()),
            )
            conn.commit()
            created = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert created is not None
            return dict(created)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def validate_execution_workspace(
        self, workspace_id: str | None, execution_id: str, role: str,
    ) -> dict:
        """Require mutating executions to own an active shadow workspace."""
        if role not in {"implementer", "repairer", "controller"}:
            return {"valid": True, "workspace": None}
        if not workspace_id:
            return {"valid": False, "reason": "mutating execution has no workspace"}
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            return {"valid": False, "reason": "workspace is not registered"}
        if workspace.get("kind") != "shadow":
            return {"valid": False, "reason": "mutating execution workspace is not a shadow"}
        if workspace.get("status") != "active":
            return {"valid": False, "reason": "mutating execution workspace is not active"}
        if workspace.get("owner_execution_id") != execution_id:
            return {"valid": False, "reason": "workspace belongs to another execution"}
        return {"valid": True, "workspace": workspace}

    def advance_canonical_workspace(
        self,
        *,
        workspace_id: str,
        expected_generation: int,
        expected_dirty_hash: str,
        applied_changeset_id: str,
        new_base_sha: str,
        new_dirty_hash: str,
    ) -> dict:
        """Advance the canonical workspace after one applied changeset."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=? AND kind='main' AND status='active'",
                (workspace_id,),
            ).fetchone()
            if row is None:
                raise WorkflowStateError("canonical workspace is not active")
            current_generation = int(row["canonical_generation"] or 0)
            current_hash = row["current_dirty_hash"] or row["dirty_patch_hash"]
            if current_generation != expected_generation or current_hash != expected_dirty_hash:
                raise WorkflowStateError("canonical workspace generation changed before advancement")
            conn.execute(
                "UPDATE workspaces SET canonical_generation=?, current_base_sha=?, "
                "current_dirty_hash=?, last_changeset_id=?, base_sha=?, dirty_patch_hash=? "
                "WHERE workspace_id=?",
                (
                    current_generation + 1, new_base_sha, new_dirty_hash,
                    applied_changeset_id, new_base_sha, new_dirty_hash, workspace_id,
                ),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert result is not None
            return dict(result)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def begin_integration_journal(
        self,
        *,
        journal_id: str,
        run_id: str,
        epoch_id: str,
        workspace_id: str,
        changeset_id: str,
        expected_generation: int,
        expected_dirty_hash: str,
    ) -> dict:
        """Open the journal for one integration, serializing by workspace.

        The DB-level canonical_generation check in advance_canonical_workspace
        runs *after* the actual ``git apply`` has already mutated the
        worktree, so it cannot by itself prevent two concurrent integrations
        against the same canonical workspace from racing at the filesystem
        level.  Rejecting a second 'applying' journal for the same
        workspace_id here -- before any git apply happens -- is what
        actually serializes them.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            in_flight = conn.execute(
                "SELECT journal_id FROM integration_journal "
                "WHERE workspace_id=? AND status='applying'",
                (workspace_id,),
            ).fetchone()
            if in_flight is not None:
                conn.rollback()
                raise WorkflowStateError(
                    f"another integration is already applying to this canonical "
                    f"workspace: {in_flight[0]}"
                )
            existing = conn.execute(
                "SELECT journal_id FROM integration_journal WHERE journal_id=?",
                (journal_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO integration_journal "
                    "(journal_id, run_id, epoch_id, workspace_id, changeset_id, "
                    "expected_generation, expected_dirty_hash, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'applying', ?)",
                    (
                        journal_id, run_id, epoch_id, workspace_id, changeset_id,
                        expected_generation, expected_dirty_hash, _utcnow(),
                    ),
                )
            else:
                # A retry of a previously terminal (failed) attempt for the
                # same changeset -- reopen it rather than silently reusing
                # the stale terminal row (INSERT OR IGNORE would have done
                # that, leaving the journal saying 'failed' while a fresh
                # git apply proceeded underneath it).
                conn.execute(
                    "UPDATE integration_journal SET run_id=?, epoch_id=?, workspace_id=?, "
                    "changeset_id=?, expected_generation=?, expected_dirty_hash=?, "
                    "status='applying', error=NULL, created_at=?, completed_at=NULL "
                    "WHERE journal_id=?",
                    (
                        run_id, epoch_id, workspace_id, changeset_id,
                        expected_generation, expected_dirty_hash, _utcnow(), journal_id,
                    ),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def finish_integration_journal(
        self, journal_id: str, status: str, error: str | None = None,
    ) -> dict | None:
        if status not in {"completed", "reconciled", "failed"}:
            raise ValueError("invalid integration journal status")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE integration_journal SET status=?, error=?, completed_at=? "
                "WHERE journal_id=? AND status='applying'",
                (status, error, _utcnow(), journal_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_pending_integration_journals(self, run_id: str | None = None) -> list[dict]:
        conn = self._new_conn()
        try:
            if run_id is None:
                rows = conn.execute(
                    "SELECT * FROM integration_journal WHERE status='applying' "
                    "ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM integration_journal WHERE status='applying' "
                    "AND run_id=? ORDER BY created_at", (run_id,)
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_workspace(self, workspace_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_workspaces(
        self,
        *,
        run_id: str | None = None,
        epoch_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        conn = self._new_conn()
        try:
            clauses: list[str] = []
            params: list[object] = []
            for column, value in (("run_id", run_id), ("epoch_id", epoch_id),
                                  ("kind", kind), ("status", status)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            query = "SELECT * FROM workspaces"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at, workspace_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def update_workspace_status(self, workspace_id: str, status: str) -> dict | None:
        if status not in {"active", "ready", "merged", "discarded", "failed"}:
            raise ValueError(f"invalid workspace status: {status}")
        conn = self._new_conn()
        try:
            released = _utcnow() if status in {"merged", "discarded", "failed"} else None
            conn.execute(
                "UPDATE workspaces SET status=?, released_at=COALESCE(?, released_at) "
                "WHERE workspace_id=?",
                (status, released, workspace_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def create_changeset(
        self,
        *,
        changeset_id: str,
        execution_id: str,
        workspace_id: str,
        base_sha: str,
        patch_digest: str,
        changed_files: list[str],
        result: dict,
        status: str,
        patch: bytes | None = None,
        parent_canonical_generation: int | None = None,
    ) -> dict:
        if status not in {"proposed", "validated", "rejected", "merged"}:
            raise ValueError(f"invalid changeset status: {status}")
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO execution_changesets
                   (changeset_id, execution_id, workspace_id, base_sha,
                    patch_digest, changed_files_json, result_json, patch_blob,
                    parent_canonical_generation, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (changeset_id, execution_id, workspace_id, base_sha, patch_digest,
                 json.dumps(changed_files, sort_keys=True),
                 json.dumps(result, sort_keys=True), patch,
                 parent_canonical_generation, status, _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_changeset(self, changeset_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_changesets(
        self,
        *,
        run_id: str | None = None,
        epoch_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[dict]:
        conn = self._new_conn()
        try:
            query = (
                "SELECT ec.* FROM execution_changesets ec "
                "JOIN workspaces w ON w.workspace_id=ec.workspace_id"
            )
            clauses: list[str] = []
            params: list[object] = []
            for expression, value in (("w.run_id", run_id), ("w.epoch_id", epoch_id),
                                      ("ec.execution_id", execution_id)):
                if value is not None:
                    clauses.append(f"{expression}=?")
                    params.append(value)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY ec.created_at, ec.changeset_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def mark_changeset_merged(self, changeset_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE execution_changesets SET status='merged' WHERE changeset_id=?",
                (changeset_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def mark_changeset_rejected(self, changeset_id: str) -> dict | None:
        """Close a changeset that the controller explicitly discarded."""
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE execution_changesets SET status='rejected' WHERE changeset_id=?",
                (changeset_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def create_integration_candidate(
        self,
        *,
        candidate_id: str,
        run_id: str,
        epoch_id: str,
        changeset_id: str,
        overlap: dict,
        validation: dict,
        disposition: str,
        integration_execution_id: str | None = None,
    ) -> dict:
        if disposition not in {"green", "yellow", "red", "pending", "resolved"}:
            raise ValueError(f"invalid integration disposition: {disposition}")
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO integration_candidates
                   (candidate_id, run_id, epoch_id, changeset_id, overlap_json,
                    validation_json, disposition, integration_execution_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, run_id, epoch_id, changeset_id,
                 json.dumps(overlap, sort_keys=True), json.dumps(validation, sort_keys=True),
                 disposition, integration_execution_id, _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def mark_integration_candidate(
        self, changeset_id: str, *, disposition: str, validation: dict | None = None,
    ) -> int:
        if disposition not in {"green", "yellow", "red", "pending", "resolved"}:
            raise ValueError(f"invalid integration disposition: {disposition}")
        conn = self._new_conn()
        try:
            sets = ["disposition=?"]
            params: list[object] = [disposition]
            if validation is not None:
                sets.append("validation_json=?")
                params.append(json.dumps(validation, sort_keys=True))
            params.append(changeset_id)
            result = conn.execute(
                f"UPDATE integration_candidates SET {', '.join(sets)} WHERE changeset_id=?",
                params,
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    def get_integration_candidates(
        self, *, run_id: str, epoch_id: str | None = None,
        disposition: str | None = None,
    ) -> list[dict]:
        """Return controller-visible changeset integration decisions."""
        conn = self._new_conn()
        try:
            clauses = ["run_id=?"]
            params: list[object] = [run_id]
            if epoch_id is not None:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            if disposition is not None:
                clauses.append("disposition=?")
                params.append(disposition)
            rows = conn.execute(
                "SELECT * FROM integration_candidates WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, candidate_id",
                params,
            ).fetchall()
            return [dict(row) for row in rows]
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

    # ---- Binding commands --------------------------------------------

    def record_binding_command(
        self,
        command_id: str,
        run_id: str,
        epoch_id: str,
        command_type: str,
        actor_type: str = "controller",
        reason: str = "",
        claude_session_id: str = "",
        claude_agent_id: str = "",
        expected_binding_version: int | None = None,
        requested_model_id: str | None = None,
        requested_role: str | None = None,
    ) -> dict:
        """Insert a new binding command in *pending* status."""
        if command_type not in VALID_COMMAND_TYPES:
            raise ValueError(f"Invalid command_type: {command_type}. Must be one of {VALID_COMMAND_TYPES}")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            conn.execute(
                """INSERT INTO binding_commands
                   (command_id, run_id, epoch_id, claude_session_id, command_type,
                    actor_type, reason, claude_agent_id, expected_binding_version,
                    requested_model_id, requested_role, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    command_id, run_id, epoch_id, claude_session_id, command_type,
                    actor_type, reason, claude_agent_id, expected_binding_version,
                    requested_model_id, requested_role, now,
                ),
            )
            conn.commit()
            return self.get_binding_command_by_id(command_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_binding_command_by_id(self, command_id: str) -> dict | None:
        """Return a command dict by its command_id."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT id, command_id, run_id, epoch_id, claude_session_id, command_type, "
                "actor_type, reason, claude_agent_id, expected_binding_version, "
                "requested_model_id, requested_role, status, applied_at, actor_id, created_at "
                "FROM binding_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("id", "command_id", "run_id", "epoch_id", "claude_session_id", "command_type",
                 "actor_type", "reason", "claude_agent_id", "expected_binding_version",
                 "requested_model_id", "requested_role", "status", "applied_at", "actor_id", "created_at"),
                row,
            ))
        finally:
            conn.close()

    def get_binding_commands(
        self, run_id: str, epoch_id: str, limit: int = 50
    ) -> list[dict]:
        """Return commands for a run/epoch ordered newest first."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT id, command_id, run_id, epoch_id, claude_session_id, command_type, "
                "actor_type, reason, claude_agent_id, expected_binding_version, "
                "requested_model_id, requested_role, status, applied_at, actor_id, created_at "
                "FROM binding_commands WHERE run_id = ? AND epoch_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (run_id, epoch_id, limit),
            ).fetchall()
            return [
                dict(zip(
                    ("id", "command_id", "run_id", "epoch_id", "claude_session_id", "command_type",
                     "actor_type", "reason", "claude_agent_id", "expected_binding_version",
                     "requested_model_id", "requested_role", "status", "applied_at", "actor_id", "created_at"),
                    row,
                ))
                for row in rows
            ]
        finally:
            conn.close()

    def apply_binding_command(
        self, command_id: str, actor_type: str = "controller", actor_id: str = ""
    ) -> dict:
        """Change a pending command to *applied* status."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id, status FROM binding_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if existing is None:
                conn.rollback()
                raise ValueError(f"Command '{command_id}' not found")
            if existing[1] != "pending":
                conn.rollback()
                raise ValueError(f"Command '{command_id}' is not pending (status: {existing[1]})")
            now = _utcnow()
            conn.execute(
                "UPDATE binding_commands SET status = 'applied', applied_at = ?, actor_type = ?, actor_id = ? "
                "WHERE command_id = ?",
                (now, actor_type, actor_id, command_id),
            )
            conn.commit()
            return self.get_binding_command_by_id(command_id)  # type: ignore[return-value]
        finally:
            conn.close()

    # ---- Workflow phases ---------------------------------------------

    def initialize_workflow_phases(
        self, run_id: str, epoch_id: str, phases: list[dict],
    ) -> list[dict]:
        """Create phase rows from a list of phase dicts (from WorkflowSpec.phases).

        Each phase dict has keys: id, roles, required, mutation, depends_on, conditional.
        Persists full phase semantics including dependencies, roles, mutation flag, and
        a specification hash for immutability verification.
        Is idempotent: if rows already exist for (run_id, epoch_id), returns existing.
        """
        import hashlib
        import json

        conn = self._new_conn()
        try:
            if not phases:
                return []

            existing = conn.execute(
                "SELECT phase_id FROM workflow_phases WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            if existing:
                return self.get_workflow_phases(run_id, epoch_id)

            for phase in phases:
                canonical = json.dumps({k: v for k, v in sorted(phase.items())}, sort_keys=True)
                spec_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                conn.execute(
                    """INSERT INTO workflow_phases
                       (run_id, epoch_id, phase_id, status, actor, required, mutating,
                       allowed_roles_json, dependencies_json, condition_json, parallel_group,
                        specification_hash, ordinal, distinct_agent_from_json, max_duration_seconds,
                        turn_budget, provider_requirements_json, min_fanout, max_fanout, result_schema,
                        quality_quorum, fallback_policy, execution_kind, required_actor,
                       max_parallelism, required_successes, max_attempts, max_attempts_per_model, sidecar_id)
                       VALUES (
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?, ?
                       )""",
                    (
                        run_id, epoch_id, phase["id"], "pending",
                        phase.get("actor", ""),
                        1 if phase.get("required", True) else 0,
                        1 if phase.get("mutation", False) else 0,
                        json.dumps(phase.get("roles", [])),
                        json.dumps(phase.get("depends_on", [])),
                        json.dumps(phase.get("conditional")) if phase.get("conditional") else None,
                        phase.get("parallel_group"),
                        spec_hash,
                        phase.get("ordinal"), json.dumps(phase.get("distinct_agent_from", [])),
                        phase.get("max_duration_seconds"), phase.get("turn_budget"),
                        json.dumps(phase.get("provider_requirements", [])), phase.get("min_fanout", 1),
                        phase.get("max_fanout", 1), phase.get("result_schema"),
                        phase.get("quality_quorum", 1), phase.get("fallback_policy"),
                        phase.get("execution_kind", "native_agent"), phase.get("actor", ""),
                        phase.get("max_parallelism"), phase.get("required_successes"),
                        phase.get("max_attempts") or phase.get("max_fanout", 1), phase.get("max_attempts_per_model"),
                        phase.get("sidecar") or phase.get("sidecar_id"),
                    ),
                )
            conn.commit()
            return self.get_workflow_phases(run_id, epoch_id)
        finally:
            conn.close()

    def get_workflow_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return all phases for an epoch, ordered by phase_id."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM workflow_phases WHERE run_id=? AND epoch_id=? ORDER BY id",
                (run_id, epoch_id),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_active_phase(self, run_id: str, epoch_id: str) -> dict | None:
        """Return the currently active phase, or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? AND status='active' ORDER BY id LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_active_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return all active phases; parallel read-only phases are valid."""
        return [
            phase for phase in self.get_workflow_phases(run_id, epoch_id)
            if phase.get("status") == "active"
        ]

    def get_ready_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return pending phases whose persisted dependencies are satisfied."""
        phases = self.get_workflow_phases(run_id, epoch_id)
        ready: list[dict] = []
        for phase in phases:
            if phase.get("status") != "pending":
                continue
            dependencies = json.loads(phase.get("dependencies_json") or "[]")
            phase_by_id = {item["phase_id"]: item for item in phases}
            if all(self._dependency_satisfied(phase_by_id.get(dep)) for dep in dependencies):
                ready.append(phase)
        return ready

    @staticmethod
    def _dependency_satisfied(phase: dict | None) -> bool:
        """Return whether a persisted phase may satisfy a dependency."""
        if phase is None:
            return False
        if phase.get("status") == "completed":
            return True
        if phase.get("status") != "skipped":
            return False
        try:
            evidence = json.loads(phase.get("result_evidence") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return evidence.get("skip_type") == "conditional"

    def advance_conditional_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Apply only deterministic conditional skips whose dependencies are ready."""
        changed: list[dict] = []
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        has_accepted_findings = any(item.get("disposition") == "accepted" for item in findings)
        for phase in self.get_ready_phases(run_id, epoch_id):
            if phase.get("condition_json") is None:
                continue
            condition = json.loads(phase["condition_json"])
            if condition == "accepted_findings" and not has_accepted_findings:
                changed.append(self.skip_conditional_phase(
                    run_id, epoch_id, phase["phase_id"],
                    "accepted_findings",
                    reason="conditional accepted_findings condition is satisfied by an empty accepted set",
                ))
        return changed

    def prepare_agent_phase(
        self, run_id: str, epoch_id: str, role: str, agent_id: str,
    ) -> dict | None:
        """Start or join the ready phase that authorizes a native agent role."""
        self.advance_conditional_phases(run_id, epoch_id)
        phases = self.get_workflow_phases(run_id, epoch_id)
        active = [
            phase for phase in phases
            if phase.get("status") == "active"
            and (
                role in json.loads(phase.get("allowed_roles_json") or "[]")
                or (role == "controller" and phase.get("actor") == "controller")
            )
        ]
        if active:
            return sorted(active, key=lambda item: (item.get("ordinal") or 0, item["phase_id"]))[0]
        ready = [
            phase for phase in self.get_ready_phases(run_id, epoch_id)
            if (
                role in json.loads(phase.get("allowed_roles_json") or "[]")
                or (role == "controller" and phase.get("actor") == "controller")
            )
            and (not phase.get("actor") or (role == "controller" and phase.get("actor") == "controller"))
        ]
        if not ready:
            raise WorkflowPhaseStateError(
                f"no ready workflow phase permits role '{role}'"
            )
        phase = sorted(ready, key=lambda item: (item.get("ordinal") or 0, item["phase_id"]))[0]
        max_duration = phase.get("max_duration_seconds")
        if max_duration and phase.get("started_at"):
            started = datetime.fromisoformat(str(phase["started_at"]))
            if (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() > max_duration:
                self.complete_phase(run_id, epoch_id, phase["phase_id"], error="phase deadline exceeded")
                raise WorkflowPhaseStateError(f"phase '{phase['phase_id']}' deadline exceeded")
        return self.start_phase(run_id, epoch_id, phase["phase_id"], actor=role)

    def complete_phase_if_ready(self, run_id: str, epoch_id: str, phase_id: str) -> dict | None:
        """Complete a phase after all assigned executions reach terminal state."""
        phase = next(
            (item for item in self.get_workflow_phases(run_id, epoch_id) if item["phase_id"] == phase_id),
            None,
        )
        if phase is None or phase.get("status") != "active":
            return phase
        executions = self.get_agent_executions(run_id, epoch_id=epoch_id, phase_id=phase_id)
        if not executions or any(
            item.get("status") in {"started", "running", "streaming", "verifying"}
            for item in executions
        ):
            return phase
        completed = [item for item in executions if item.get("status") == "completed"]
        if phase.get("result_schema"):
            successful = [
                item for item in completed
                if item.get("schema_valid") is True
                and item.get("evidence_valid") is not False
                and item.get("accepted_by_controller") is not False
            ]
        else:
            successful = [
                item for item in completed
                if item.get("evidence_valid") is not False
                and item.get("accepted_by_controller") is not False
            ]
        required_successes = int(
            phase.get("required_successes")
            or max(int(phase.get("min_fanout") or 1), int(phase.get("quality_quorum") or 1))
        )
        if len(successful) >= required_successes:
            return self.complete_phase(
                run_id, epoch_id, phase_id,
                result_evidence=json.dumps({
                    "execution_ids": [item["execution_id"] for item in executions],
                    "completed": len(completed),
                    "successful": len(successful),
                }, separators=(",", ":")),
            )

        failed = [item for item in executions if item.get("status") != "completed"]
        max_attempts = int(phase.get("max_attempts") or phase.get("max_fanout") or 1)
        fallback_policy = str(phase.get("fallback_policy") or "").strip()
        if fallback_policy and len(executions) < max_attempts:
            # Keep the phase active so the scheduler can materialize the next
            # retry/fallback action.  Failure evidence remains in the ledger.
            return phase
        if failed or len(executions) >= max_attempts:
            first_failure = failed[0] if failed else executions[-1]
            return self.complete_phase(
                run_id, epoch_id, phase_id,
                error=f"execution failure: {first_failure.get('execution_id')}",
            )
        return phase

    def start_phase(
        self, run_id: str, epoch_id: str, phase_id: str, actor: str = "",
        principal: str = "",
    ) -> dict:
        """Start a phase: set status='active', record started_at.

        All dependency and actor semantics are read from the immutable
        persisted phase instance.  Callers cannot provide a replacement
        phase definition at transition time.
        """
        conn = self._new_conn()
        try:
            # Check phase exists and is pending
            row = conn.execute(
                "SELECT id, status, dependencies_json, required_actor, mutating "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[1] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[1]}, cannot start (must be 'pending')"
                )
            required_actor = str(row[3] or "")
            if required_actor and _canonical_actor(actor) != _canonical_actor(required_actor):
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' requires actor '{required_actor}', got '{actor}'"
                )
            if row[4]:
                active_mutation = conn.execute(
                    "SELECT phase_id FROM workflow_phases WHERE run_id=? AND epoch_id=? "
                    "AND status='active' AND mutating=1 LIMIT 1", (run_id, epoch_id)
                ).fetchone()
                if active_mutation is not None:
                    raise WorkflowPhaseStateError(
                        f"mutating phase '{active_mutation[0]}' is already active"
                    )

            # Dependencies come from the immutable persisted phase snapshot;
            # caller-supplied definitions cannot bypass the DAG.
            dependencies = json.loads(row[2] or "[]")
            phase_rows = conn.execute(
                "SELECT phase_id, status, result_evidence FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            phase_by_id = {str(item[0]): dict(item) for item in phase_rows}
            unsatisfied = [
                dependency for dependency in dependencies
                if not self._dependency_satisfied(phase_by_id.get(dependency))
            ]
            if unsatisfied:
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' dependencies are not satisfied: {', '.join(unsatisfied)}"
                )

            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='active', started_at=?, actor=?, "
                "started_by_actor=?, started_by_principal=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, actor, actor, principal or None, run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def complete_phase(
        self, run_id: str, epoch_id: str, phase_id: str,
        result_evidence: str = "", error: str = "",
    ) -> dict:
        """Complete a phase: set status='completed' (or 'failed' if error given).

        Only 'active' phases can be completed. Auto-sets status to 'failed' when
        error is non-empty.
        """
        new_status = "failed" if error else "completed"
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] != "active":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot complete (must be 'active')"
                )

            phase_row = conn.execute(
                "SELECT allowed_roles_json, min_fanout, max_fanout, quality_quorum, "
                "started_at, distinct_agent_from_json, max_duration_seconds, result_schema, "
                "turn_budget, max_attempts FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?", (run_id, epoch_id, phase_id)
            ).fetchone()
            if phase_row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if new_status == "completed" and not result_evidence.strip():
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' requires result_evidence before completion"
                )
            if new_status == "completed" and phase_row[6] and phase_row[4]:
                started = datetime.fromisoformat(str(phase_row[4]))
                elapsed = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
                if elapsed > int(phase_row[6]):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded its {phase_row[6]} second deadline"
                    )
            if new_status == "completed" and phase_row[7]:
                try:
                    schema = json.loads(str(phase_row[7]))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_schema is invalid JSON"
                    ) from exc
                try:
                    parsed_evidence = json.loads(result_evidence)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence is not valid JSON"
                    ) from exc
                required_keys = schema.get("required", []) if isinstance(schema, dict) else []
                if not isinstance(parsed_evidence, dict) or not isinstance(required_keys, list):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence does not match its schema"
                    )
                missing = sorted(str(key) for key in required_keys if key not in parsed_evidence)
                if missing:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence is missing: {', '.join(missing)}"
                    )
            executions = conn.execute(
                "SELECT claude_agent_id, independence_key, status, tool_call_count, "
                "schema_valid, evidence_valid, accepted_by_controller, quality_score "
                "FROM agent_executions "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?", (run_id, epoch_id, phase_id)
            ).fetchall()
            if new_status == "completed" and phase_row[8] and executions:
                observed_turns = sum(int(item[3] or 0) for item in executions)
                if observed_turns > int(phase_row[8]):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded turn budget {phase_row[8]}"
                    )
            if new_status == "completed" and executions:
                completed = [item for item in executions if item[2] == "completed"]
                if phase_row[7]:
                    quality_eligible = [
                        item for item in completed
                        if item[4] == 1 and item[5] != 0 and item[6] != 0
                    ]
                else:
                    quality_eligible = [
                        item for item in completed
                        if item[5] != 0 and item[6] != 0
                    ]
                min_fanout = int(phase_row[1] or 1)
                max_fanout = int(phase_row[2] or 1)
                quorum = int(phase_row[3] or 1)
                max_attempts = int(phase_row[9] or max_fanout)
                if len(executions) > max_attempts:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded max attempts {max_attempts}"
                    )
                if len(completed) < min_fanout or len(quality_eligible) < quorum:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' requires {max(min_fanout, quorum)} quality-valid execution(s)"
                    )
                distinct_refs = json.loads(phase_row[5] or "[]")
                if distinct_refs:
                    prior = conn.execute(
                        "SELECT claude_agent_id, independence_key FROM agent_executions WHERE run_id=? AND epoch_id=? "
                        "AND phase_id IN (%s)" % ",".join("?" * len(distinct_refs)),
                        [run_id, epoch_id, *distinct_refs],
                    ).fetchall()
                    prior_keys = {item[1] for item in prior if item[1]}
                    prior_ids = {item[0] for item in prior if not item[1]}
                    if any(
                        (item[1] and item[1] in prior_keys)
                        or (not item[1] and item[0] in prior_ids)
                        for item in completed
                    ):
                        raise WorkflowPhaseStateError(
                            f"Phase '{phase_id}' violates distinct_agent_from"
                        )
            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status=?, completed_at=?, result_evidence=?, error=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (new_status, now, result_evidence, error, run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def skip_phase(
        self, run_id: str, epoch_id: str, phase_id: str, reason: str = "",
    ) -> dict:
        """Skip a non-required phase: set status='skipped'.

        Raises WorkflowPhaseStateError if phase is required.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status, required FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot skip (must be 'pending')"
                )
            if bool(row[1]):
                raise WorkflowPhaseStateError(
                    f"Required phase '{phase_id}' cannot be skipped"
                )

            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='skipped', completed_at=?, result_evidence=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, json.dumps({"skip_type": "manual", "reason": reason}),
                 run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def validate_phase_transition(
        self, run_id: str, epoch_id: str, target_phase_id: str,
        phases_spec: list[dict] | None = None,
    ) -> dict:
        """Validate that *target_phase_id* can be started given current phase state.

        ``phases_spec`` is retained as a compatibility argument for older
        callers, but it is intentionally ignored.  Persisted phase instances
        are the only authority for dependencies and transition state.

        Returns dict: {"valid": True} or {"valid": False, "reason": "..."}

        Rules:
        1. Target phase must exist in phases_spec
        2. Target phase must be 'pending' (not already started/completed/skipped)
        3. All phases in depends_on must be 'completed'
        4. If a required dependency is 'failed', target cannot start
        """
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT phase_id, status, dependencies_json, result_evidence "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            phase_rows = {str(row[0]): dict(row) for row in rows}
        finally:
            conn.close()

        # Check target is pending
        target = phase_rows.get(target_phase_id)
        if target is None:
            return {"valid": False, "reason": f"Phase '{target_phase_id}' has no state record"}
        if target["status"] != "pending":
            return {"valid": False, "reason": f"Phase '{target_phase_id}' is '{target['status']}', not 'pending'"}

        # Check dependencies
        for dep_id in json.loads(target.get("dependencies_json") or "[]"):
            dependency = phase_rows.get(dep_id)
            if dependency is None:
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' has no state record"}
            if dependency["status"] == "failed":
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' failed, cannot proceed"}
            if not self._dependency_satisfied(dependency):
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' is not satisfied"}

        return {"valid": True}

    def ensure_workflow_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Ensure workflow phases exist for the given run+epoch.

        Loads the WorkflowSpec from the registry (by workflow_id) and
        creates phase rows if they don't yet exist.
        """
        existing = self.get_workflow_phases(run_id, epoch_id)
        if existing:
            return existing

        # Get the epoch to find the workflow_id
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            return []

        workflow_id = active.get("workflow_id", "")
        if not workflow_id:
            return []

        # Load spec from registry
        from enhanced_router.registry import get_registry

        registry = get_registry()
        spec = registry.get_workflow(workflow_id)
        if spec is None:
            return []

        phases_data = [
            {
                "id": p.id,
                "roles": p.roles,
                "required": p.required,
                "mutation": p.mutation,
                "depends_on": p.depends_on,
                "conditional": p.conditional,
                "actor": p.actor or "",
                "distinct_agent_from": p.distinct_agent_from,
                "parallel_group": p.parallel_group,
                "ordinal": p.ordinal,
                "max_duration_seconds": p.max_duration_seconds,
                "turn_budget": p.turn_budget,
                "provider_requirements": p.provider_requirements,
                "min_fanout": p.min_fanout,
                "max_fanout": p.max_fanout,
                "result_schema": p.result_schema,
                "quality_quorum": p.quality_quorum,
                "fallback_policy": p.fallback_policy,
                "execution_kind": p.execution_kind,
            }
            for p in spec.phases
        ]

        return self.initialize_workflow_phases(run_id, epoch_id, phases_data)

    # ---- Condition evaluation for conditional phases ---------------------

    def evaluate_condition(
        self, run_id: str, epoch_id: str, phase_id: str, condition: str | None,
    ) -> dict:
        """Evaluate whether a conditional phase should be activated or skipped.

        Supported conditions:

        - ``accepted_findings``: Requires at least one open accepted finding.
          If none exist, the phase is auto-skipped (not required).
        - ``None`` or empty: Condition is satisfied (no constraint).

        Returns dict with keys: satisfied (bool), reason (str), evidence (dict).
        """
        if not condition:
            return {"satisfied": True, "reason": "No condition", "evidence": {}}

        if condition == "accepted_findings":
            findings = self.get_open_accepted_findings(run_id, epoch_id)
            count = len(findings)
            if count > 0:
                return {
                    "satisfied": True,
                    "reason": f"{count} accepted open finding(s) require repair",
                    "evidence": {"accepted_finding_count": count},
                }
            return {
                "satisfied": False,
                "reason": "No accepted findings to repair — auto-skipping",
                "evidence": {"accepted_finding_count": 0},
            }

        return {
            "satisfied": False,
            "reason": f"Unknown condition '{condition}'",
            "evidence": {},
        }

    def skip_conditional_phase(
        self, run_id: str, epoch_id: str, phase_id: str, condition: str | None,
        reason: str = "",
    ) -> dict:
        """Evaluate and skip a conditional phase if its condition is not met.

        Returns the phase dict (with status 'skipped' if skipped,
        or unchanged status if condition is satisfied).
        """
        evaluation = self.evaluate_condition(run_id, epoch_id, phase_id, condition)
        if evaluation["satisfied"]:
            # Condition is satisfied — do not skip, phase stays pending
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)

        # Auto-skip: condition is not satisfied
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] == "skipped":
                phases = self.get_workflow_phases(run_id, epoch_id)
                return next(p for p in phases if p["phase_id"] == phase_id)
            if row[0] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot skip (must be 'pending')"
                )

            required = conn.execute(
                "SELECT required FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if required is not None and bool(required[0]):
                raise WorkflowPhaseStateError(
                    f"Required phase '{phase_id}' cannot be conditionally skipped"
                )
            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='skipped', completed_at=?, result_evidence=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, json.dumps({
                    "skip_type": "conditional",
                    "condition": condition,
                    "reason": reason or evaluation.get("reason", ""),
                    "evidence": evaluation.get("evidence", {}),
                }),
                 run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()


# ---- Finding lifecycle -----------------------------------------------


    # ---- Agent execution lifecycle ---------------------------------------

    def create_agent_execution(
        self,
        execution_id: str,
        run_id: str,
        epoch_id: str,
        claude_agent_id: str,
        role: str,
        model_id: str,
        *,
        phase_id: str | None = None,
        binding_id: int | None = None,
        actor_kind: str = "subagent",
        execution_kind: str = "subagent",
        provider_id: str | None = None,
        endpoint_id: str | None = None,
        transport: str | None = None,
        configuration_hash: str | None = None,
        parent_execution_id: str | None = None,
        workspace_id: str | None = None,
        independence_key: str | None = None,
    ) -> dict:
        """Record the start of an authoritative execution.

        When a binding is supplied, all identity fields are checked against
        it. Hooks cannot create a record that claims a different model or
        role than the router actually bound.
        """
        conn = self._new_conn()
        try:
            if role in {"implementer", "repairer", "controller"}:
                if not workspace_id:
                    raise ValueError("mutating agent execution requires a shadow workspace")
                workspace_row = conn.execute(
                    "SELECT kind, status, owner_execution_id FROM workspaces WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchone()
                if (
                    workspace_row is None
                    or workspace_row[0] != "shadow"
                    or workspace_row[1] != "active"
                    or workspace_row[2] != execution_id
                ):
                    raise ValueError(
                        "mutating agent execution must own an active shadow workspace"
                    )
            # A native lifecycle hook can run before the first model request
            # creates a binding.  When an active binding already exists, use
            # it as the authoritative identity instead of trusting caller
            # supplied model/provider fields.
            if binding_id is None:
                existing_binding = self._select_agent_binding(
                    conn, run_id, claude_agent_id,
                )
                if existing_binding is not None:
                    binding_id = int(existing_binding["binding_id"])

            if binding_id is not None:
                binding = conn.execute(
                    "SELECT run_id, epoch_id, claude_agent_id, role, model_id, released_at, "
                    "provider_id, endpoint_id, configuration_hash "
                    "FROM agent_bindings WHERE binding_id=?", (binding_id,)
                ).fetchone()
                if binding is None:
                    raise ValueError(f"unknown agent binding {binding_id}")
                if binding[0] != run_id or binding[1] != epoch_id or binding[2] != claude_agent_id:
                    raise ValueError("agent execution does not match its binding scope")
                if binding[3] != role or binding[4] != model_id:
                    raise ValueError("agent execution identity does not match its binding")
                if binding[5] is not None:
                    raise ValueError("agent binding is already released")
                if provider_id is not None and binding[6] is not None and provider_id != binding[6]:
                    raise ValueError("agent execution provider does not match its binding")
                if endpoint_id is not None and binding[7] is not None and endpoint_id != binding[7]:
                    raise ValueError("agent execution endpoint does not match its binding")
                if (
                    configuration_hash is not None
                    and binding[8] is not None
                    and configuration_hash != binding[8]
                ):
                    raise ValueError("agent execution configuration does not match its binding")

            if independence_key is None:
                independence_key = hashlib.sha256(
                    json.dumps({
                        "model_id": model_id,
                        "provider_id": provider_id,
                        "endpoint_id": endpoint_id,
                        "role": role,
                        "phase_id": phase_id,
                        "parent_execution_id": parent_execution_id,
                    }, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()

            if phase_id is not None:
                phase = conn.execute(
                    "SELECT status, required_actor, allowed_roles_json, max_fanout, "
                    "distinct_agent_from_json, provider_requirements_json, "
                    "max_duration_seconds, turn_budget, max_parallelism, max_attempts "
                    "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()
                if phase is None:
                    raise ValueError(f"unknown workflow phase {phase_id!r}")
                if phase[0] != "active":
                    raise ValueError(
                        f"workflow phase {phase_id!r} is {phase[0]}, not active"
                    )
                if phase[6] is not None:
                    started_at = conn.execute(
                        "SELECT started_at FROM workflow_phases "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (run_id, epoch_id, phase_id),
                    ).fetchone()[0]
                    if started_at:
                        elapsed = (
                            datetime.now(timezone.utc)
                            - datetime.fromisoformat(str(started_at)).astimezone(timezone.utc)
                        ).total_seconds()
                        if elapsed > int(phase[6]):
                            raise ValueError(
                                f"workflow phase {phase_id!r} exceeded its deadline"
                            )
                provider_requirements = json.loads(phase[5] or "[]")
                if provider_requirements and provider_id not in provider_requirements:
                    raise ValueError(
                        f"provider {provider_id!r} is not permitted by workflow phase {phase_id!r}"
                    )
                allowed_roles = json.loads(phase[2] or "[]")
                actor = str(phase[1] or "")
                if role not in allowed_roles and not (role == "controller" and actor == "controller"):
                    raise ValueError(
                        f"role {role!r} is not allowed in workflow phase {phase_id!r}"
                    )
                if actor and actor not in {role, "controller"}:
                    raise ValueError(
                        f"workflow phase {phase_id!r} requires actor {actor!r}"
                    )
                active_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=? "
                    "AND status NOT IN ('completed','failed','timeout','cancelled')",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                attempt_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                if int(active_count) >= int(phase[8] or phase[3] or 1):
                    raise ValueError(
                        f"workflow phase {phase_id!r} exceeded max fanout/parallelism"
                    )
                if int(attempt_count) >= int(phase[9] or phase[3] or 1):
                    raise ValueError(f"workflow phase {phase_id!r} exceeded max attempts")
                if phase[7] is not None:
                    used_turns = conn.execute(
                        "SELECT COALESCE(SUM(tool_call_count), 0) FROM agent_executions "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (run_id, epoch_id, phase_id),
                    ).fetchone()[0]
                    if int(used_turns or 0) >= int(phase[7]):
                        raise ValueError(
                            f"workflow phase {phase_id!r} exhausted turn budget {phase[7]}"
                        )
                distinct_refs = json.loads(phase[4] or "[]")
                if distinct_refs:
                    placeholders = ",".join("?" for _ in distinct_refs)
                    prior = conn.execute(
                        "SELECT claude_agent_id, independence_key FROM agent_executions "
                        f"WHERE run_id=? AND epoch_id=? AND phase_id IN ({placeholders})",
                        [run_id, epoch_id, *distinct_refs],
                    ).fetchall()
                    if any(
                        (independence_key and row[1] == independence_key)
                        or (not row[1] and row[0] == claude_agent_id)
                        for row in prior
                    ):
                        raise ValueError(
                            f"agent {claude_agent_id!r} violates distinct_agent_from "
                            f"for workflow phase {phase_id!r}"
                        )
            conn.execute(
                """INSERT INTO agent_executions
                   (execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                    phase_id, binding_id, status, actor_kind, execution_kind, provider_id,
                    endpoint_id, transport, configuration_hash, parent_execution_id,
                    workspace_id, independence_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                 phase_id, binding_id, actor_kind, execution_kind, provider_id,
                 endpoint_id, transport, configuration_hash, parent_execution_id,
                 workspace_id, independence_key),
            )
            conn.commit()
            return self.get_agent_execution(execution_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_agent_execution(self, execution_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_agent_execution_scoped(
        self, run_id: str, epoch_id: str, execution_id: str,
    ) -> dict | None:
        """Return an execution only when both run and epoch own it."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=? "
                "AND run_id=? AND epoch_id=?",
                (execution_id, run_id, epoch_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_agent_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        result_type: str | None = None,
        result_summary: str | None = None,
        output_hash: str | None = None,
        error: str | None = None,
        tool_call_count: int | None = None,
        total_tokens: int | None = None,
        result_json: str | None = None,
        request_count: int | None = None,
        retry_count: int | None = None,
        input_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        output_tokens: int | None = None,
        ttft_ms: float | None = None,
        wall_time_ms: float | None = None,
        error_class: str | None = None,
        schema_valid: bool | None = None,
        evidence_valid: bool | None = None,
        accepted_by_controller: bool | None = None,
        quality_score: float | None = None,
        verdict: str | None = None,
        confidence: float | None = None,
    ) -> dict | None:
        """Update an agent execution while enforcing lifecycle transitions.

        ``timed_out`` is accepted as the public spelling and stored as the
        legacy schema's ``timeout`` value.  Metrics are deliberately written
        to ``error_class``; the human-readable ``error`` field is preserved.
        """
        conn = self._new_conn()
        try:
            current = conn.execute(
                "SELECT status FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if current is None:
                return None
            sets = ["updated_at = datetime('now')"]
            params: list = []
            if status is not None:
                normalized_status = "timeout" if status == "timed_out" else status
                if normalized_status not in _EXECUTION_STATUSES:
                    raise ValueError(f"invalid agent execution status: {status}")
                current_status = str(current[0])
                if current_status in _EXECUTION_TERMINAL_STATUSES:
                    if normalized_status != current_status:
                        raise WorkflowStateError(
                            f"terminal execution cannot transition from {current_status!r} "
                            f"to {normalized_status!r}"
                        )
                elif normalized_status not in _EXECUTION_TRANSITIONS.get(current_status, frozenset()):
                    raise WorkflowStateError(
                        f"illegal execution transition {current_status!r} -> {normalized_status!r}"
                    )
                sets.append("status = ?")
                params.append(normalized_status)
                if normalized_status in _EXECUTION_TERMINAL_STATUSES:
                    sets.append("completed_at = datetime('now')")
            if result_type is not None:
                sets.append("result_type = ?")
                params.append(result_type)
            if result_summary is not None:
                sets.append("result_summary = ?")
                params.append(result_summary)
            if output_hash is not None:
                sets.append("output_hash = ?")
                params.append(output_hash)
            if error is not None:
                sets.append("error = ?")
                params.append(error)
            if tool_call_count is not None:
                sets.append("tool_call_count = ?")
                params.append(tool_call_count)
            if total_tokens is not None:
                sets.append("total_tokens = ?")
                params.append(total_tokens)
            for column, value in (
                ("result_json", result_json), ("request_count", request_count),
                ("retry_count", retry_count), ("input_tokens", input_tokens),
                ("cache_read_tokens", cache_read_tokens), ("cache_write_tokens", cache_write_tokens),
                ("output_tokens", output_tokens), ("ttft_ms", ttft_ms),
                ("wall_time_ms", wall_time_ms), ("error_class", error_class),
                ("schema_valid", None if schema_valid is None else int(schema_valid)),
                ("evidence_valid", None if evidence_valid is None else int(evidence_valid)),
                ("accepted_by_controller", None if accepted_by_controller is None else int(accepted_by_controller)),
                ("quality_score", quality_score), ("verdict", verdict),
                ("confidence", confidence),
            ):
                if value is not None:
                    sets.append(f"{column} = ?")
                    params.append(value)
            params.append(execution_id)
            conn.execute(
                f"UPDATE agent_executions SET {', '.join(sets)} WHERE execution_id = ?",
                params,
            )
            conn.commit()
            return self.get_agent_execution(execution_id)
        finally:
            conn.close()

    def increment_execution_tool_calls(
        self, execution_id: str, *, run_id: str | None = None,
        epoch_id: str | None = None, delta: int = 1,
    ) -> dict | None:
        """Atomically increment tool calls; caller counters are not trusted."""
        if delta < 1:
            raise ValueError("tool-call increment must be positive")
        conn = self._new_conn()
        try:
            clauses = ["execution_id=?"]
            params: list[object] = [execution_id]
            if run_id is not None:
                clauses.append("run_id=?")
                params.append(run_id)
            if epoch_id is not None:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            conn.execute(
                "UPDATE agent_executions SET tool_call_count=COALESCE(tool_call_count, 0)+?, "
                "updated_at=datetime('now') WHERE " + " AND ".join(clauses),
                [delta, *params],
            )
            conn.commit()
            return self.get_agent_execution(execution_id)
        finally:
            conn.close()

    def increment_execution_requests(self, binding_id: int | None) -> dict | None:
        """Correlate an admitted model request with its execution ledger row."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE agent_executions SET request_count=COALESCE(request_count, 0)+1, "
                "updated_at=datetime('now') WHERE binding_id=?",
                (binding_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE binding_id=? "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_execution_metrics_for_binding(
        self,
        binding_id: int | None,
        *,
        input_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        output_tokens: int = 0,
        ttft_ms: float | None = None,
        wall_time_ms: float | None = None,
    ) -> dict | None:
        """Merge non-sensitive request metrics into the bound execution."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE agent_executions SET input_tokens=COALESCE(input_tokens, 0)+?, "
                "cache_read_tokens=COALESCE(cache_read_tokens, 0)+?, "
                "cache_write_tokens=COALESCE(cache_write_tokens, 0)+?, "
                "output_tokens=COALESCE(output_tokens, 0)+?, "
                "ttft_ms=COALESCE(ttft_ms, ?), wall_time_ms=COALESCE(?, wall_time_ms), "
                "updated_at=datetime('now') WHERE binding_id=?",
                (input_tokens, cache_read_tokens, cache_write_tokens, output_tokens,
                 ttft_ms, wall_time_ms, binding_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE binding_id=? "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_execution_failure_for_binding(
        self, binding_id: int | None, *, error_class: str, error: str,
    ) -> dict | None:
        """Persist a transport failure against the active bound execution."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT execution_id, run_id, epoch_id FROM agent_executions "
                "WHERE binding_id=? AND status IN ('started','running') "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,),
            ).fetchone()
            if row is None:
                return None
            now = _utcnow()
            conn.execute(
                "UPDATE agent_executions SET status='failed', completed_at=?, error=?, "
                "error_class=?, updated_at=? WHERE execution_id=? AND status IN ('started','running')",
                (now, error[:500], error_class, now, row[0]),
            )
            conn.commit()
            result = self.get_agent_execution(str(row[0]))
            try:
                self.append_execution_event(
                    str(row[1]), str(row[2]), str(row[0]), "failed",
                    {"error_class": error_class, "reason": error[:500]},
                )
            except Exception:
                logger.debug("failed to append transport failure event", exc_info=True)
            return result
        finally:
            conn.close()

    def get_agent_executions(
        self,
        run_id: str,
        epoch_id: str | None = None,
        phase_id: str | None = None,
        role: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List agent executions with optional filters."""
        conn = self._new_conn()
        try:
            parts = ["SELECT * FROM agent_executions WHERE run_id = ?"]
            params: list = [run_id]
            if epoch_id:
                parts.append("AND epoch_id = ?")
                params.append(epoch_id)
            if phase_id:
                parts.append("AND phase_id = ?")
                params.append(phase_id)
            if role:
                parts.append("AND role = ?")
                params.append(role)
            if status:
                parts.append("AND status = ?")
                params.append(status)
            parts.append("ORDER BY started_at DESC")
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


    # ---- Finding lifecycle -----------------------------------------------

    def create_finding(
        self,
        finding_id: str,
        run_id: str,
        epoch_id: str,
        description: str,
        *,
        severity: str = "medium",
        category: str = "",
        source_phase_id: str | None = None,
        source_agent_id: str | None = None,
        evidence_json: str | None = None,
    ) -> dict:
        """Record a new finding. Returns the created finding row."""
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO findings
                   (finding_id, run_id, epoch_id, severity, category, description,
                    source_phase_id, source_agent_id, evidence_json, disposition, verification_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'pending')""",
                (finding_id, run_id, epoch_id, severity, category, description,
                 source_phase_id, source_agent_id, evidence_json or "{}"),
            )
            conn.commit()
            return self.get_finding(finding_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_finding(self, finding_id: str) -> dict | None:
        """Return a single finding by finding_id."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM findings WHERE finding_id = ?", (finding_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_finding_scoped(
        self, run_id: str, epoch_id: str, finding_id: str,
    ) -> dict | None:
        """Return a finding only when it belongs to the requested epoch."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM findings WHERE finding_id=? AND run_id=? AND epoch_id=?",
                (finding_id, run_id, epoch_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_findings(
        self,
        run_id: str,
        epoch_id: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List findings for a run, optionally filtered by epoch, disposition, or status."""
        conn = self._new_conn()
        try:
            parts = ["SELECT * FROM findings WHERE run_id = ?"]
            params: list[str] = [run_id]
            if epoch_id:
                parts.append("AND epoch_id = ?")
                params.append(epoch_id)
            if disposition:
                parts.append("AND disposition = ?")
                params.append(disposition)
            if status:
                parts.append("AND verification_status = ?")
                params.append(status)
            parts.append("ORDER BY created_at DESC")
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def adjudicate_finding(
        self,
        finding_id: str,
        disposition: str,
        *,
        reason: str = "",
        dispositioned_by: str = "",
        repair_agent_id: str | None = None,
        repair_phase_id: str | None = None,
    ) -> dict | None:
        """Accept, reject, waive, or mark a finding as duplicate.

        Accepted findings with a repair_agent_id are flagged for resolution.
        """
        conn = self._new_conn()
        try:
            conn.execute(
                """UPDATE findings SET disposition=?, disposition_reason=?,
                   dispositioned_at=datetime('now'), dispositioned_by=?,
                   repair_agent_id=?, repair_phase_id=?, updated_at=datetime('now')
                   WHERE finding_id=?""",
                (disposition, reason, dispositioned_by, repair_agent_id,
                 repair_phase_id, finding_id),
            )
            conn.commit()
            return self.get_finding(finding_id)
        finally:
            conn.close()

    def resolve_finding(
        self,
        finding_id: str,
        verification_status: str,
        resolution_evidence_json: str = "{}",
    ) -> dict | None:
        """Record resolution evidence and verification result for a finding."""
        conn = self._new_conn()
        try:
            conn.execute(
                """UPDATE findings SET verification_status=?,
                   resolution_evidence_json=?, updated_at=datetime('now')
                   WHERE finding_id=?""",
                (verification_status, resolution_evidence_json, finding_id),
            )
            conn.commit()
            return self.get_finding(finding_id)
        finally:
            conn.close()

    def get_open_accepted_findings(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return findings that are accepted but not yet verified (for repair contracts)."""
        return self.get_findings(run_id, epoch_id=epoch_id, disposition="accepted", status="pending")


# ------------------------------------------------------------------ Singleton

_state: RouteState | None = None


def get_state() -> RouteState:
    global _state
    if _state is None:
        _state = RouteState(DEFAULT_DB_PATH)
    return _state
