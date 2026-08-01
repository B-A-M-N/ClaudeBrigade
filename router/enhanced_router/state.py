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
from enhanced_router.state_errors import WorkflowStateError
from enhanced_router.litellm_state import LiteLLMGenerationRepository
from enhanced_router.model_health_state import ModelHealthRepository
from enhanced_router.mutation_lease_state import MutationLeaseRepository
from enhanced_router.finding_state import FindingRepository
from enhanced_router.route_state import RouteOperationsRepository
from enhanced_router.binding_state import BindingRepository
from enhanced_router.epoch_state import EpochRepository
from enhanced_router.controller_binding_state import ControllerBindingRepository
from enhanced_router.cas_route_state import CasRouteRepository, RouteConflictError
from enhanced_router.controller_policy_state import ControllerPolicyRepository, ControllerModelError
from enhanced_router.binding_command_state import BindingCommandRepository, VALID_COMMAND_TYPES
from enhanced_router.workflow_phase_state import WorkflowPhaseRepository, WorkflowPhaseStateError
from enhanced_router.condition_evaluation_state import ConditionEvaluationRepository
from enhanced_router.agent_execution_state import AgentExecutionRepository
from enhanced_router.shadow_workspace_state import ShadowWorkspaceRepository
from enhanced_router.intake_fastpath_state import IntakeFastpathRepository
from enhanced_router.provider_reservation_state import ProviderReservationRepository
from enhanced_router.sidecar_execution_state import SidecarExecutionRepository
from enhanced_router.native_spawn_attach_state import NativeSpawnAttachRepository
from enhanced_router.run_registry_state import RunRegistryRepository
from enhanced_router.runnable_action_state import RunnableActionRepository, _utcnow_age
from enhanced_router.run_orchestration_state import RunOrchestrationRepository

logger = logging.getLogger("claude-enhanced-router")

# Keep the public schema version stable for existing Brigade installations;
# the hardening migration below is shape-detected so databases already at v36
# are upgraded without forcing a second version bump.
SCHEMA_VERSION = 36

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))


def _canonical_actor(actor: str | None) -> str:
    """Normalize legacy ``<role>-agent`` labels before exact comparison."""
    value = str(actor or "").strip().lower()
    if value.endswith("-agent"):
        return value[:-len("-agent")]
    return value


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
    if current < 38:
        _migrate_v38(conn)
        conn.execute("PRAGMA user_version = 38")
    if current < 39:
        _migrate_v39(conn)
        conn.execute("PRAGMA user_version = 39")
    if current < 40:
        _migrate_v40(conn)
        conn.execute("PRAGMA user_version = 40")


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


def _migrate_v38(conn: sqlite3.Connection) -> None:
    """Persist each run's own launch selection.

    The launcher exported CLAUDE_BRIGADE_PROFILE as a process environment
    variable, which only reaches the single controller session that inherited
    it -- a shared router daemon serving multiple concurrent runs has no way
    to know which profile a given hook invocation's run actually selected.
    Persisting the selection on the run row itself makes it authoritative and
    queryable independent of the launching process's environment.
    """
    _add_column_if_missing(conn, "runs", "inference_profile_id", "TEXT")
    _add_column_if_missing(conn, "runs", "sidecar_profile_id", "TEXT")
    _add_column_if_missing(conn, "runs", "launch_preset_id", "TEXT")


def _migrate_v39(conn: sqlite3.Connection) -> None:
    """Add fallback_routes_json: same fallback ladder as fallback_models_json
    but each entry is a {"model", "endpoint"} dict instead of a bare model ID,
    so a fallback can pin its own endpoint/provider instead of always
    inheriting the primary route's endpoint (or "auto").
    """
    _add_column_if_missing(conn, "role_routes", "fallback_routes_json", "TEXT")


def _migrate_v40(conn: sqlite3.Connection) -> None:
    """Give route_proposals a real queued/running/completed/failed/expired
    lifecycle bound to the run/epoch it was generated for.

    Previously the row was only INSERTed after the detached fastpath model
    call finished, so a proposal_id handed to a caller in the "queued" HTTP
    response could resolve to "not found" for however long inference took.
    It also had no run_id/epoch_id of its own (only reachable indirectly via
    its intake), so a late proposal from an earlier epoch could be accepted
    during a later epoch of the same run. status/expires_at/configuration_hash/
    candidate_digest support reserving the row up front and CAS-guarding
    every subsequent transition (running/completed/failed/disposition).
    Existing rows all predate this lifecycle and were only ever written once
    fully resolved, so they default to status='completed'.
    """
    _add_column_if_missing(conn, "route_proposals", "run_id", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "epoch_id", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "execution_id", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "status", "TEXT NOT NULL DEFAULT 'completed'")
    _add_column_if_missing(conn, "route_proposals", "configuration_hash", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "candidate_digest", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "expires_at", "TEXT")
    _add_column_if_missing(conn, "route_proposals", "completed_at", "TEXT")


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

class RouteState(
    LiteLLMGenerationRepository, ModelHealthRepository, MutationLeaseRepository, FindingRepository,
    RouteOperationsRepository, BindingRepository, EpochRepository,
    ControllerBindingRepository, CasRouteRepository, ControllerPolicyRepository,
    BindingCommandRepository, WorkflowPhaseRepository, ConditionEvaluationRepository,
    AgentExecutionRepository, ShadowWorkspaceRepository, IntakeFastpathRepository,
    ProviderReservationRepository, SidecarExecutionRepository, NativeSpawnAttachRepository,
    RunRegistryRepository, RunnableActionRepository, RunOrchestrationRepository,
):
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

    # create_task_intake, create_route_proposal, get_route_proposal,
    # get_task_intake, get_route_proposal_for_run,
    # set_route_proposal_disposition, create_fastpath_verification live in
    # IntakeFastpathRepository (intake_fastpath_state.py), mixed in below.

    # reserve_provider_agent, provider_agent_capacity_available,
    # admit_provider_agents, release_provider_reservation,
    # get_provider_reservations, attach_provider_reservation,
    # release_all_provider_reservations live in ProviderReservationRepository
    # (provider_reservation_state.py), mixed in below.


    # _active_action_claims, get_runnable_actions, claim_runnable_action,

    # consume_controller_action, finish_controller_action,

    # cancel_action_claims, reconcile_lifecycle, create_spawn_intent,

    # update_spawn_intent (and _role_route_fallback_candidates) live in

    # RunnableActionRepository (runnable_action_state.py), mixed in below.




















    # ---- Run lifecycle ---------------------------------------------

    # begin_task, validate_completion, prepare_completion_token,

    # consume_completion_token live in RunOrchestrationRepository

    # (run_orchestration_state.py), mixed in below.





    # ---- Epoch lifecycle -------------------------------------------
    # get_active_epoch, create_epoch, close_epoch, set_profile_routes_atomic,
    # create_epoch_from_profile live in EpochRepository (epoch_state.py),
    # mixed in below.

    # ---- Route operations ------------------------------------------
    # set_role_route, get_role_route, get_epoch_routes (plus the
    # _ROLE_ROUTE_COLUMNS/_row_to_role_route helpers) live in
    # RouteOperationsRepository (route_state.py), mixed in below.

    # ---- Binding operations ----------------------------------------
    # _select_agent_binding, bind_or_get_agent, get_agent_binding,
    # release_binding, get_active_bindings live in BindingRepository
    # (binding_state.py), mixed in below.

    # ---- Controller bindings and endpoint observations ----------------
    # _select_controller_binding, get_controller_binding,
    # bind_or_get_controller, record_endpoint_usage, is_endpoint_certified,
    # record_endpoint_certifications, get_endpoint_certifications,
    # get_endpoint_observations live in ControllerBindingRepository
    # (controller_binding_state.py), mixed in below.

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
    # set_model_health, get_model_health, set_model_certification,
    # get_model_certification live in ModelHealthRepository
    # (model_health_state.py), mixed in below.

    # ---- LiteLLM catalog lifecycle -----------------------------------
    # create_litellm_generation, activate_litellm_generation,
    # get_active_litellm_generation, register_litellm_deployment,
    # update_litellm_deployment, get_litellm_deployment,
    # get_active_litellm_deployment, get_litellm_deployment_for_generation
    # live in LiteLLMGenerationRepository (litellm_state.py), mixed in below.

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
    # set_role_route_cas (and RouteConflictError) live in CasRouteRepository
    # (cas_route_state.py), mixed in below; RouteConflictError is
    # re-exported from this module's imports so `from enhanced_router.state
    # import RouteConflictError` keeps working.

    # ---- Controller policy -------------------------------------------
    # upsert_controller_policy, get_controller_policy,
    # validate_controller_model (and ControllerModelError) live in
    # ControllerPolicyRepository (controller_policy_state.py), mixed in
    # below; ControllerModelError is re-exported from this module's imports
    # so `from enhanced_router.state import ControllerModelError` keeps
    # working.

    # ---- Mutation leases ---------------------------------------------
    # acquire_mutation_lease, release_mutation_lease,
    # heartbeat_mutation_lease, get_mutation_lease,
    # get_active_mutation_leases, get_active_mutator live in
    # MutationLeaseRepository (mutation_lease_state.py), mixed in below.

    # ---- Shadow workspace / changeset lifecycle ------------------------

    # create_workspace, register_main_workspace, validate_execution_workspace,
    # advance_canonical_workspace, begin_integration_journal,
    # finish_integration_journal, get_pending_integration_journals,
    # get_workspace, get_workspaces, update_workspace_status, create_changeset,
    # get_changeset, get_changesets, mark_changeset_merged,
    # mark_changeset_rejected, create_integration_candidate,
    # mark_integration_candidate, get_integration_candidates live in
    # ShadowWorkspaceRepository (shadow_workspace_state.py), mixed in below.
    # expire_stale_leases moved to MutationLeaseRepository
    # (mutation_lease_state.py) -- it operates on mutation_leases, not
    # workspaces, and was only ever grouped here by proximity.

    # ---- Binding commands --------------------------------------------
    # record_binding_command, get_binding_command_by_id, get_binding_commands,
    # apply_binding_command (and VALID_COMMAND_TYPES) live in
    # BindingCommandRepository (binding_command_state.py), mixed in below;
    # VALID_COMMAND_TYPES is re-exported from this module's imports so
    # `from enhanced_router.state import VALID_COMMAND_TYPES` keeps working.

    # ---- Workflow phases ---------------------------------------------

    # initialize_workflow_phases, get_workflow_phases, get_active_phase,
    # get_active_phases, get_ready_phases, _dependency_satisfied,
    # advance_conditional_phases, prepare_agent_phase, complete_phase_if_ready,
    # start_phase, complete_phase, skip_phase, validate_phase_transition,
    # ensure_workflow_phases (and WorkflowPhaseStateError) live in
    # WorkflowPhaseRepository (workflow_phase_state.py), mixed in below;
    # WorkflowPhaseStateError is re-exported from this module's imports so
    # `from enhanced_router.state import WorkflowPhaseStateError` keeps
    # working.

    # ---- Condition evaluation for conditional phases ---------------------
    # evaluate_condition, skip_conditional_phase live in
    # ConditionEvaluationRepository (condition_evaluation_state.py), mixed
    # in below.

    # ---- Agent execution lifecycle ---------------------------------------

    # create_agent_execution, get_agent_execution, get_agent_execution_scoped,
    # update_agent_execution, increment_execution_tool_calls,
    # increment_execution_requests, record_execution_metrics_for_binding,
    # record_execution_failure_for_binding, get_agent_executions live in
    # AgentExecutionRepository (agent_execution_state.py), mixed in below.


    # ---- Finding lifecycle -----------------------------------------------
    # create_finding, get_finding, get_finding_scoped, get_findings,
    # adjudicate_finding, resolve_finding, get_open_accepted_findings live
    # in FindingRepository (finding_state.py), mixed in below.


# ------------------------------------------------------------------ Singleton

_state: RouteState | None = None


def get_state() -> RouteState:
    global _state
    if _state is None:
        _state = RouteState(DEFAULT_DB_PATH)
    return _state
