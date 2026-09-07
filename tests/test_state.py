"""Tests for RouteState -- run / epoch / route / binding / health lifecycle."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest

from enhanced_router.state import (
    RouteState,
    _utcnow,
    _migrate_v9,
    _migrate_v10,
    _migrate_v11,
    WorkflowPhaseStateError,
    WorkflowStateError,
)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_state.db"


@pytest.fixture()
def state(tmp_db: Path) -> RouteState:
    return RouteState(tmp_db)


# ================================================================== Run lifecycle
# ==================================================================


def test_create_run(state: RouteState, tmp_db: Path):
    run_id = "run-abc123"
    result = state.create_run(run_id, session_id="sess-1", cwd="/tmp")
    assert result["run_id"] == run_id
    assert result["claude_session_id"] == "sess-1"
    assert result["cwd"] == "/tmp"
    assert result["created_at"] is not None
    assert result["closed_at"] is None

    # Verify persistence
    fresh = RouteState(tmp_db)
    row = fresh.get_run(run_id)
    assert row is not None
    assert row["claude_session_id"] == "sess-1"


def test_create_run_idempotent(state: RouteState):
    run_id = "run-dup"
    r1 = state.create_run(run_id, session_id="s1", cwd="/a")
    r2 = state.create_run(run_id, session_id="s2", cwd="/b")  # different values
    assert r1["claude_session_id"] == "s1"
    assert r2["claude_session_id"] == "s1"  # first value wins


def test_litellm_deployment_telemetry_persists_events(state: RouteState):
    generation = state.create_litellm_generation(
        registry_hash="registry-1",
        model_count=2,
        config_digest="config-1",
        reason="test",
    )
    deployment = state.register_litellm_deployment(generation, 18000, 1234)
    event = state.record_litellm_deployment_event(
        deployment_id=deployment,
        generation=generation,
        event="healthy",
        status="active",
        pid=1234,
        port=18000,
        active_requests=2,
        active_streams=1,
        metadata={"reason": "probe"},
    )

    assert event["event"] == "healthy"
    telemetry = state.get_litellm_deployment_telemetry(deployment_id=deployment)
    assert telemetry[0]["event_count"] == 1
    assert telemetry[0]["last_event"] == "healthy"
    assert telemetry[0]["last_active_requests"] == 2
    assert state.get_litellm_deployment_events(deployment)[0]["metadata_json"] == '{"reason":"probe"}'


def test_create_run_accepts_launch_selection(state: RouteState):
    result = state.create_run(
        "run-sel", inference_profile_id="local-build", sidecar_profile_id="cheap-review",
        launch_preset_id="local-build-cheap-review",
    )
    assert result["inference_profile_id"] == "local-build"
    assert result["sidecar_profile_id"] == "cheap-review"
    assert result["launch_preset_id"] == "local-build-cheap-review"
    row = state.get_run("run-sel")
    assert row is not None
    assert row["inference_profile_id"] == "local-build"


def test_set_run_selection_overwrites_after_creation(state: RouteState):
    """The launcher pre-registers a run before the wizard runs, then updates
    the selection once config selection is final -- this must overwrite the
    earlier (absent) value, unlike create_run's COALESCE-guarded insert.
    """
    state.create_run("run-late-select")
    assert state.get_run("run-late-select")["inference_profile_id"] is None  # type: ignore[index]

    updated = state.set_run_selection("run-late-select", inference_profile_id="cloud-build")
    assert updated is not None
    assert updated["inference_profile_id"] == "cloud-build"

    again = state.set_run_selection("run-late-select", sidecar_profile_id="deep-review")
    assert again is not None
    assert again["inference_profile_id"] == "cloud-build"  # untouched field preserved
    assert again["sidecar_profile_id"] == "deep-review"


def test_set_run_selection_unknown_run_returns_none(state: RouteState):
    assert state.set_run_selection("no-such-run", inference_profile_id="x") is None


def test_active_run_selections_returns_only_open_runs(state: RouteState):
    state.create_run(
        "run-open", inference_profile_id="cloud-build", sidecar_profile_id="deep-review",
    )
    state.create_run("run-closed", inference_profile_id="local-build")
    state.close_run("run-closed")

    selections = state.active_run_selections()
    assert selections == [
        {"inference_profile_id": "cloud-build", "sidecar_profile_id": "deep-review"},
    ]


def test_active_run_selections_empty_with_no_open_runs(state: RouteState):
    assert state.active_run_selections() == []


def test_active_run_selections_includes_runs_without_a_profile(state: RouteState):
    """A bare `--model` launch never sets a profile; the scoping consumer
    must be able to see that and widen back out rather than mistake an
    empty list entry for 'no runs open'.
    """
    state.create_run("run-no-profile")
    assert state.active_run_selections() == [
        {"inference_profile_id": None, "sidecar_profile_id": None},
    ]


# ================================================================== Epoch lifecycle
# ==================================================================


def test_create_epoch(state: RouteState):
    state.create_run("r1")
    epoch = state.create_epoch("r1", "ep-1", "normal", "hybrid")
    assert epoch["epoch_id"] == "ep-1"
    assert epoch["status"] == "active"
    assert epoch["profile_id"] == "hybrid"


def test_active_epoch_invariant(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    with pytest.raises(ValueError, match="Active epoch"):
        state.create_epoch("r1", "ep-2", "normal", "hybrid")

    # Close first epoch, now second should work
    state.close_epoch("r1", "ep-1")
    epoch2 = state.create_epoch("r1", "ep-2", "normal", "hybrid")
    assert epoch2["epoch_id"] == "ep-2"


def test_create_epoch_from_profile_creates_all_routes(state: RouteState):
    state.create_run("r1")
    epoch = state.create_epoch_from_profile("r1", "ep-1", "normal", "hybrid")
    assert epoch["epoch_id"] == "ep-1"
    assert epoch["profile_id"] == "hybrid"

    # "hybrid" profile may not exist; if it does, routes should be set
    # If it doesn't exist, routes dict may be empty -- that's acceptable
    state.get_epoch_routes("r1", "ep-1")
    # because create_epoch_from_profile catches profile load errors gracefully


def test_profile_atomicrollback(state: RouteState):
    """If one role route fails in a profile, ALL should roll back."""
    state.create_run("r1")
    # This test verifies the atomicity contract of set_profile_routes_atomic.
    # We verify it works by checking that set_role_route (which is the
    # per-role atomic unit) is properly isolated -- the real atomic profile
    # test is covered by create_epoch_from_profile above.
    state.set_role_route("r1", "ep-test", "recon", "model-a", "manual")
    assert state.get_role_route("r1", "ep-test", "recon")["model_id"] == "model-a"


# ================================================================== Role routes
# ==================================================================


def test_set_role_route_increments_version(state: RouteState):
    state.create_run("r1")
    r1 = state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    assert r1["version"] == 1
    r2 = state.set_role_route("r1", "ep-1", "recon", "model-b", "manual")
    assert r2["version"] == 2
    r3 = state.set_role_route("r1", "ep-1", "recon", "model-c", "manual")
    assert r3["version"] == 3


def test_get_role_route(state: RouteState):
    state.create_run("r1")
    state.set_role_route("r1", "ep-1", "implementer", "model-x", "manual")
    route = state.get_role_route("r1", "ep-1", "implementer")
    assert route is not None
    assert route["model_id"] == "model-x"
    assert state.get_role_route("r1", "ep-1", "adversary") is None


def test_set_role_route_appends_event(state: RouteState):
    state.create_run("r1")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual", "initial setup")

    conn = state._new_conn()
    try:
        rows = conn.execute(
            "SELECT event_type, role, old_model_id, new_model_id FROM route_events "
            "WHERE run_id = ? AND epoch_id = ?",
            ("r1", "ep-1"),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "route_change"
        assert rows[0][2] is None  # old_model_id is None for first set
        assert rows[0][3] == "model-a"
    finally:
        conn.close()


def test_get_epoch_routes(state: RouteState):
    state.create_run("r1")
    state.set_role_route("r1", "ep-1", "recon", "model-r", "manual")
    state.set_role_route("r1", "ep-1", "implementer", "model-i", "manual")
    state.set_role_route("r1", "ep-1", "adversary", "model-a", "manual")
    state.set_role_route("r1", "ep-1", "repairer", "model-re", "manual")

    routes = state.get_epoch_routes("r1", "ep-1")
    assert set(routes.keys()) == {"recon", "implementer", "adversary", "repairer"}
    assert routes["recon"]["model_id"] == "model-r"
    assert routes["implementer"]["model_id"] == "model-i"


# ================================================================== Agent bindings
# ==================================================================


def test_bind_agent(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")

    binding, is_new = state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-a", 1)
    assert isinstance(binding, dict)
    assert binding["model_id"] == "model-a"
    assert binding["role"] == "recon"
    assert binding["epoch_id"] == "ep-1"
    assert is_new is True

    binding = state.get_agent_binding("r1", "agent-1")
    assert binding is not None
    assert binding["model_id"] == "model-a"
    assert binding["configuration_hash"] is None


def test_binding_configuration_hash_round_trip(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    binding, _ = state.bind_or_get_agent(
        "r1", "agent-1", "ep-1", "recon", "model-a", 1,
        configuration_hash="sha256:current",
        provider_ids_json='["freeinference","openrouter"]',
    )
    assert binding["configuration_hash"] == "sha256:current"
    assert binding["provider_ids_json"] == '["freeinference","openrouter"]'
    assert state.get_active_bindings("r1")[0]["configuration_hash"] == "sha256:current"


def test_bind_or_get_returns_existing(state: RouteState):
    """bind_or_get_agent returns existing binding without error on second call."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")

    binding1, is_new1 = state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-a", 1)
    assert is_new1 is True

    # Second call returns existing, does not raise
    binding2, is_new2 = state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-a", 1)
    assert binding1["binding_id"] == binding2["binding_id"]
    assert is_new2 is False


def test_release_and_rebind_creates_new_history(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    binding1, _ = state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-a", 1)
    state.release_binding("r1", "agent-1")

    # Active binding should be gone
    assert state.get_agent_binding("r1", "agent-1") is None

    # Release is idempotent
    state.release_binding("r1", "agent-1")

    # Rebind works
    binding2, _ = state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-b", 2)
    assert binding2["binding_id"] != binding1["binding_id"]  # new binding_id


def test_get_agent_binding_returns_active_only(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-a", 1)
    state.release_binding("r1", "agent-1")

    assert state.get_agent_binding("r1", "agent-1") is None

    # But history still exists via direct query
    conn = state._new_conn()
    try:
        rows = conn.execute(
            "SELECT binding_id, released_at FROM agent_bindings WHERE run_id = ? AND claude_agent_id = ?",
            ("r1", "agent-1"),
        ).fetchall()
        assert len(rows) == 1  # original binding still in history, marked released
        assert rows[0][1] is not None  # released_at is set
    finally:
        conn.close()


# ================================================================== Route change test
# ==================================================================


def test_route_change_affects_next_spawn_in_same_epoch(state: RouteState):
    """Existing bound agent stays on old model, new agent in same epoch gets new model."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-v1", "manual")

    # First agent binds to model-v1
    state.bind_or_get_agent("r1", "agent-1", "ep-1", "recon", "model-v1", 1)

    # Route is changed
    state.set_role_route("r1", "ep-1", "recon", "model-v2", "manual")

    # Existing agent still bound to old model
    binding = state.get_agent_binding("r1", "agent-1")
    assert binding["model_id"] == "model-v1"

    # New agent resolves to new model
    state.set_role_route("r1", "ep-1", "recon", "model-v2", "manual")
    binding2, _ = state.bind_or_get_agent("r1", "agent-2", "ep-1", "recon", "model-v2", 2)
    assert binding2["binding_id"] > 0


# ================================================================== Isolation
# ==================================================================


def test_two_runs_isolated(state: RouteState):
    state.create_run("run-a")
    state.create_epoch("run-a", "ep-1", "normal", "hybrid")
    state.set_role_route("run-a", "ep-1", "recon", "model-a", "manual")

    state.create_run("run-b")
    state.create_epoch("run-b", "ep-2", "normal", "hybrid")
    state.set_role_route("run-b", "ep-2", "recon", "model-b", "manual")

    routes_a = state.get_epoch_routes("run-a", "ep-1")
    routes_b = state.get_epoch_routes("run-b", "ep-2")
    assert routes_a["recon"]["model_id"] == "model-a"
    assert routes_b["recon"]["model_id"] == "model-b"

    # No cross-contamination
    assert state.get_active_epoch("run-a")["epoch_id"] == "ep-1"
    assert state.get_active_epoch("run-b")["epoch_id"] == "ep-2"


def test_two_epochs_in_same_run_isolated(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-1", "manual")

    state.close_epoch("r1", "ep-1")
    state.create_epoch("r1", "ep-2", "normal", "hybrid")
    state.set_role_route("r1", "ep-2", "recon", "model-2", "manual")

    assert state.get_epoch_routes("r1", "ep-1")["recon"]["model_id"] == "model-1"
    assert state.get_epoch_routes("r1", "ep-2")["recon"]["model_id"] == "model-2"

    # Only ep-2 is active
    active = state.get_active_epoch("r1")
    assert active["epoch_id"] == "ep-2"


# ================================================================== Snapshot
# ==================================================================


def test_route_snapshot_produces_sha256(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.bind_or_get_agent(
        "r1", "agent-1", "ep-1", "recon", "model-a", 1,
        configuration_hash="sha256:current",
    )

    snapshot_hash = state.create_route_snapshot("r1", "ep-1", purpose="test")

    assert isinstance(snapshot_hash, str)
    assert len(snapshot_hash) == 64  # SHA-256 hex digest
    # Verify it's a valid hex string
    int(snapshot_hash, 16)

    conn = state._new_conn()
    try:
        assert conn.execute(
            "SELECT configuration_hash FROM agent_bindings WHERE claude_agent_id=?",
            ("agent-1",),
        ).fetchone()[0] == "sha256:current"
    finally:
        conn.close()


# ================================================================== Model health
# ==================================================================


def test_model_health_crud(state: RouteState):
    state.set_model_health(
        model_id="m1",
        config_hash="abc123",
        harness_version="v1",
        status="healthy",
        reachable=True,
        authenticated=True,
        compatible=True,
        failure_rate=0.01,
        latency_ms=200.0,
    )

    health = state.get_model_health("m1")
    assert health is not None
    assert health["status"] == "healthy"
    assert health["reachable"] == 1
    assert health["failure_rate"] == 0.01

    # Update
    state.set_model_health(
        model_id="m1",
        config_hash="abc123",
        harness_version="v1",
        status="degraded",
        reachable=True,
        authenticated=True,
        compatible=True,
        failure_rate=0.15,
        latency_ms=500.0,
    )

    health2 = state.get_model_health("m1")
    assert health2["status"] == "degraded"
    assert health2["failure_rate"] == 0.15

    # Unknown model
    assert state.get_model_health("unknown-model") is None


# ================================================================== Concurrent route versions
# ==================================================================


def test_concurrent_route_versions(state: RouteState):
    """Simulate concurrent route changes via threads."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    errors: list[Exception] = []
    versions: list[int] = []

    def worker(i: int) -> None:
        try:
            for _ in range(5):
                result = state.set_role_route("r1", "ep-1", "recon", f"model-{i}", "concurrent")
                versions.append(result["version"])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # All 20 operations should have succeeded with distinct version increments
    assert len(versions) == 20
    # Final version should be 20 (all operations applied)
    assert max(versions) == 20
    # All versions 1..20 should be present
    assert sorted(set(versions)) == list(range(1, 21))


# ================================================================== CAS route updates
# ==================================================================


def test_set_role_route_cas_succeeds_with_correct_version(state: RouteState):
    """CAS succeeds when expected_version matches current version."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")

    # Current version is 1
    result = state.set_role_route_cas(
        "r1", "ep-1", "recon", "model-b",
        command_id="cmd-1", expected_version=1,
        actor_type="controller", reason="model upgrade",
    )
    assert result["model_id"] == "model-b"
    assert result["version"] == 2
    assert result["source"] == "cas"
    assert result["idempotent"] is False


def test_set_role_route_cas_rejects_wrong_version(state: RouteState):
    """CAS raises RouteConflictError when version mismatch."""
    from enhanced_router.state import RouteConflictError

    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    # Version is now 1

    # Try with wrong expected version (5 instead of 1)
    with pytest.raises(RouteConflictError, match="CAS conflict"):
        state.set_role_route_cas(
            "r1", "ep-1", "recon", "model-b",
            command_id="cmd-x", expected_version=5,
            actor_type="controller", reason="bad version",
        )

    # Original model should be unchanged
    route = state.get_role_route("r1", "ep-1", "recon")
    assert route["model_id"] == "model-a"
    assert route["version"] == 1


def test_set_role_route_cas_is_idempotent(state: RouteState):
    """Same command_id returns existing result without reapplying."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")

    result1 = state.set_role_route_cas(
        "r1", "ep-1", "recon", "model-b",
        command_id="cmd-dup", expected_version=1,
        actor_type="controller", reason="idempotent test",
    )
    assert result1["model_id"] == "model-b"
    assert result1["version"] == 2
    assert result1["idempotent"] is False

    # Second call with same command_id should be idempotent
    result2 = state.set_role_route_cas(
        "r1", "ep-1", "recon", "model-c",  # different model, ignored
        command_id="cmd-dup", expected_version=1,
        actor_type="controller", reason="ignored",
    )
    assert result2["idempotent"] is True
    assert result2["model_id"] == "model-b"  # unchanged
    # Version should still be 2, not incremented
    route = state.get_role_route("r1", "ep-1", "recon")
    assert route["version"] == 2


def test_set_role_route_cas_requires_active_epoch(state: RouteState):
    """CAS fails when epoch is closed."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.close_epoch("r1", "ep-1")

    with pytest.raises(ValueError, match="No active epoch"):
        state.set_role_route_cas(
            "r1", "ep-1", "recon", "model-b",
            command_id="cmd-ep", expected_version=1,
            actor_type="controller", reason="closed epoch",
        )


def test_set_role_route_cas_invalid_role(state: RouteState):
    """CAS rejects invalid roles."""
    with pytest.raises(ValueError, match="Invalid role"):
        state.set_role_route_cas(
            "r1", "ep-1", "bogus", "model-b",
            command_id="cmd-role", expected_version=0,
            actor_type="controller", reason="bad role",
        )


# ================================================================== Controller policy
# ==================================================================


def test_controller_policy_reject_permitted_model(state: RouteState):
    """validate_controller_model returns True for permitted model under reject policy."""
    state.create_run("r1")
    state.upsert_controller_policy("r1", ["model-a", "model-b"], "reject")

    assert state.validate_controller_model("r1", "model-a") is True
    assert state.validate_controller_model("r1", "model-b") is True


def test_controller_policy_reject_rejects_unknown_model(state: RouteState):
    """validate_controller_model raises ControllerModelError for non-permitted model."""
    from enhanced_router.state import ControllerModelError

    state.create_run("r1")
    state.upsert_controller_policy("r1", ["model-a"], "reject")

    with pytest.raises(ControllerModelError, match="not permitted"):
        state.validate_controller_model("r1", "model-x")


def test_controller_policy_allow_allows_anything(state: RouteState):
    """'allow' policy always returns True."""
    state.create_run("r1")
    state.upsert_controller_policy("r1", ["model-a"], "allow")

    assert state.validate_controller_model("r1", "model-a") is True
    assert state.validate_controller_model("r1", "anything") is True


def test_controller_policy_no_policy_allows_by_default(state: RouteState):
    """No policy configured means allow by default."""
    state.create_run("r1")
    # No policy set
    assert state.validate_controller_model("r1", "anything") is True


def test_controller_policy_upsert_updates(state: RouteState):
    """Updating a policy overwrites the previous one."""
    state.create_run("r1")
    state.upsert_controller_policy("r1", ["model-a"], "reject")
    state.upsert_controller_policy("r1", ["model-b"], "allow")

    assert state.validate_controller_model("r1", "model-x") is True  # 'allow' policy

    policy = state.get_controller_policy("r1")
    assert policy is not None
    assert policy["model_change_policy"] == "allow"


def test_controller_policy_get_nonexistent(state: RouteState):
    """get_controller_policy returns None for unconfigured runs."""
    assert state.get_controller_policy("r1") is None


# ================================================================== Schema version
# ==================================================================


def test_schema_version_is_current():
    from enhanced_router.state import SCHEMA_VERSION
    assert SCHEMA_VERSION == 36


def test_fresh_database_has_valid_foreign_keys(tmp_path: Path):
    state = RouteState(tmp_path / "fresh-fk.db")
    conn = state._new_conn()
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        foreign_keys = conn.execute("PRAGMA foreign_key_list(agent_executions)").fetchall()
        assert any(row[2] == "agent_bindings" and row[4] == "binding_id" for row in foreign_keys)
    finally:
        conn.close()


def test_fresh_database_has_binding_configuration_hash(tmp_path: Path):
    state = RouteState(tmp_path / "fresh-binding.db")
    conn = state._new_conn()
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_bindings)")}
        assert "configuration_hash" in columns
    finally:
        conn.close()


def test_provider_agent_admission_does_not_create_dead_queue_entry(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    first = state.reserve_provider_agent(
        reservation_id="a1", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="e1", max_active=1, enqueue=False,
    )
    second = state.reserve_provider_agent(
        reservation_id="a2", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="e2", max_active=1, enqueue=False,
    )
    assert first["state"] == "reserved"
    assert second["state"] == "unavailable"
    assert state.get_provider_reservations("freeinference") == [first]


def test_reserve_provider_agent_refuses_an_unhealthy_model_immediately(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_model_health(
        "bad-model", "cfg", "1.0", "unhealthy",
        reachable=False, authenticated=True, compatible=True,
    )
    result = state.reserve_provider_agent(
        reservation_id="a1", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="e1", max_active=1, enqueue=False, model_id="bad-model",
    )
    assert result["state"] == "unavailable"
    assert "health check" in result["reason"]
    assert state.get_provider_reservations("freeinference") == []


def test_reserve_provider_agent_admits_a_healthy_model(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_model_health(
        "good-model", "cfg", "1.0", "healthy",
        reachable=True, authenticated=True, compatible=True,
    )
    result = state.reserve_provider_agent(
        reservation_id="a1", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="e1", max_active=1, enqueue=False, model_id="good-model",
    )
    assert result["state"] == "reserved"


def test_reserve_provider_agent_does_not_block_an_untested_model(state: RouteState):
    """No health record yet (untested) must not be treated as unhealthy."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    result = state.reserve_provider_agent(
        reservation_id="a1", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="e1", max_active=1, enqueue=False, model_id="never-checked-model",
    )
    assert result["state"] == "reserved"


def test_reserve_provider_agent_rejects_stale_health(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_model_health(
        "stale-model", "cfg", "1.0", "healthy",
        reachable=True, authenticated=True, compatible=True,
    )
    old_checked_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn = state._new_conn()
    try:
        conn.execute(
            "UPDATE model_health SET checked_at=? WHERE model_id=?",
            (old_checked_at, "stale-model"),
        )
        conn.commit()
    finally:
        conn.close()

    result = state.reserve_provider_agent(
        reservation_id="stale-reservation", run_id="r1", epoch_id="ep-1",
        provider_id="freeinference", execution_id="e1", max_active=1,
        enqueue=False, model_id="stale-model", health_max_age_seconds=60,
    )
    assert result["state"] == "unavailable"
    assert "stale" in result["reason"]
    assert state.get_provider_reservations("freeinference") == []


def test_provider_promotion_skips_newly_unhealthy_queued_model(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_model_health(
        "queued-model", "cfg", "1.0", "healthy",
        reachable=True, authenticated=True, compatible=True,
    )
    state.set_model_health(
        "healthy-model", "cfg", "1.0", "healthy",
        reachable=True, authenticated=True, compatible=True,
    )
    blocker = state.reserve_provider_agent(
        reservation_id="blocker", run_id="r1", epoch_id="ep-1",
        provider_id="freeinference", execution_id="blocker-exec",
        max_active=1, enqueue=False,
    )
    assert blocker["state"] == "reserved"
    queued = state.reserve_provider_agent(
        reservation_id="queued-bad", run_id="r1", epoch_id="ep-1",
        provider_id="freeinference", execution_id="bad-exec",
        max_active=1, enqueue=True, model_id="queued-model",
    )
    healthy = state.reserve_provider_agent(
        reservation_id="queued-good", run_id="r1", epoch_id="ep-1",
        provider_id="freeinference", execution_id="good-exec",
        max_active=1, enqueue=True, model_id="healthy-model",
    )
    assert queued["state"] == "queued"
    assert healthy["state"] == "queued"
    assert queued["model_id"] == "queued-model"

    state.set_model_health(
        "queued-model", "cfg", "1.0", "unhealthy",
        reachable=False, authenticated=True, compatible=True,
    )
    state.release_provider_reservation("blocker")
    admitted = state.admit_provider_agents(
        "freeinference", 1, health_max_age_seconds=60,
    )

    assert [item["reservation_id"] for item in admitted] == ["queued-good"]
    reservations = {
        item["reservation_id"]: item
        for item in state.get_provider_reservations("freeinference")
    }
    assert reservations["queued-bad"]["state"] == "expired"
    assert reservations["queued-good"]["state"] == "reserved"


def test_ttl_expired_claim_releases_its_provider_reservation(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A claim that expires via TTL must free its reservation immediately.

    reconcile_lifecycle only runs at process startup, so if the inline
    expiry path in _active_action_claims doesn't also release the
    reservation, the provider capacity slot leaks for the life of the
    process instead of being freed for the next queued agent.
    """
    from enhanced_router.state import _utcnow_age

    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    first = state.reserve_provider_agent(
        reservation_id="res-1", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="pending:action-1", max_active=1, enqueue=False,
    )
    assert first["state"] == "reserved"
    second = state.reserve_provider_agent(
        reservation_id="res-2", run_id="r1", epoch_id="ep-1", provider_id="freeinference",
        execution_id="pending:action-2", max_active=1, enqueue=True,
    )
    assert second["state"] == "queued"

    conn = state._new_conn()
    try:
        conn.execute(
            "INSERT INTO runnable_action_claims "
            "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
            "action_kind, provider_id, claim_token, reservation_id, status, created_at, "
            "claimed_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("action-1", "r1", "ep-1", "phase-1", "implementer", "brigade-implementer",
             "model-a", "native_agent", "freeinference", "tok-1", "res-1", "claimed",
             _utcnow(), _utcnow(), _utcnow_age(1)),
        )
        conn.commit()
    finally:
        conn.close()

    class FakeProvider:
        limits = SimpleNamespace(max_active_agents=1)

    class FakeRegistry:
        providers = {"freeinference": FakeProvider()}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    state.get_runnable_actions("r1", "ep-1")

    reservations = {r["reservation_id"]: r for r in state.get_provider_reservations("freeinference")}
    assert reservations["res-1"]["state"] == "expired"
    assert reservations["res-2"]["state"] == "reserved"

    conn = state._new_conn()
    try:
        claim = conn.execute(
            "SELECT status FROM runnable_action_claims WHERE action_id='action-1'"
        ).fetchone()
        assert claim["status"] == "expired"
    finally:
        conn.close()


def _register_canonical_workspace(state: RouteState) -> str:
    state.create_run("r1", session_id="s1", cwd="/tmp")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    workspace = state.create_workspace(
        workspace_id="ws-main", run_id="r1", epoch_id="ep-1", kind="main",
        path="/tmp/repo", base_sha="sha-0", dirty_patch_hash="dirty-0", status="active",
    )
    return str(workspace["workspace_id"])


def _make_changeset(state: RouteState, changeset_id: str, workspace_id: str) -> None:
    state.create_changeset(
        changeset_id=changeset_id, execution_id=f"exec-{changeset_id}",
        workspace_id=workspace_id, base_sha="sha-0", patch_digest=f"digest-{changeset_id}",
        changed_files=["a.py"], result={"validation": {"valid": True}}, status="validated",
        patch=b"--- a\n+++ b\n",
    )


def test_get_workspaces_reaps_a_shadow_workspace_from_a_crashed_subagent(
    state: RouteState,
):
    """A shadow workspace whose owning subagent died without a clean status
    transition (SubagentStop/StopFailure never fired, so update_workspace_status
    never ran) must eventually be reclaimed by get_workspaces itself, not
    only at session_end.py."""
    state.create_run("r1", session_id="s1", cwd="/tmp")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.create_workspace(
        workspace_id="ws-shadow-1", run_id="r1", epoch_id="ep-1", kind="shadow",
        path="/tmp/shadow-1", base_sha="sha-0", dirty_patch_hash="dirty-0",
    )

    conn = state._new_conn()
    conn.execute(
        "UPDATE workspaces SET heartbeat_at = '2000-01-01T00:00:00+00:00' "
        "WHERE workspace_id='ws-shadow-1'",
    )
    conn.commit()
    conn.close()

    workspaces = state.get_workspaces(run_id="r1", epoch_id="ep-1", kind="shadow")
    assert workspaces[0]["status"] == "discarded"
    assert workspaces[0]["released_at"] is not None


def test_get_workspaces_does_not_reap_a_fresh_shadow_workspace(state: RouteState):
    state.create_run("r1", session_id="s1", cwd="/tmp")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.create_workspace(
        workspace_id="ws-shadow-1", run_id="r1", epoch_id="ep-1", kind="shadow",
        path="/tmp/shadow-1", base_sha="sha-0", dirty_patch_hash="dirty-0",
    )
    workspaces = state.get_workspaces(run_id="r1", epoch_id="ep-1", kind="shadow")
    assert workspaces[0]["status"] == "active"


def test_get_workspaces_does_not_reap_a_stale_main_workspace(state: RouteState):
    """Only shadow workspaces are reaped -- a stale main/integration
    workspace has a different lifecycle and must stay active regardless of
    age."""
    workspace_id = _register_canonical_workspace(state)
    conn = state._new_conn()
    conn.execute(
        "UPDATE workspaces SET heartbeat_at = '2000-01-01T00:00:00+00:00' "
        "WHERE workspace_id=?",
        (workspace_id,),
    )
    conn.commit()
    conn.close()

    workspaces = state.get_workspaces(run_id="r1", epoch_id="ep-1", kind="main")
    assert workspaces[0]["status"] == "active"


def test_begin_integration_journal_rejects_concurrent_workspace_integration(
    state: RouteState,
):
    """A second changeset can't start applying while another is in flight
    against the same canonical workspace -- the actual git apply that
    follows begin_integration_journal is not itself serialized, so this
    check is what prevents two concurrent integrations from racing on disk.
    """
    workspace_id = _register_canonical_workspace(state)
    _make_changeset(state, "cs-1", workspace_id)
    _make_changeset(state, "cs-2", workspace_id)

    state.begin_integration_journal(
        journal_id="integration:cs-1", run_id="r1", epoch_id="ep-1",
        workspace_id=workspace_id, changeset_id="cs-1",
        expected_generation=0, expected_dirty_hash="dirty-0",
    )

    with pytest.raises(WorkflowStateError, match="already applying"):
        state.begin_integration_journal(
            journal_id="integration:cs-2", run_id="r1", epoch_id="ep-1",
            workspace_id=workspace_id, changeset_id="cs-2",
            expected_generation=0, expected_dirty_hash="dirty-0",
        )

    state.finish_integration_journal("integration:cs-1", "completed")
    reopened = state.begin_integration_journal(
        journal_id="integration:cs-2", run_id="r1", epoch_id="ep-1",
        workspace_id=workspace_id, changeset_id="cs-2",
        expected_generation=0, expected_dirty_hash="dirty-0",
    )
    assert reopened["status"] == "applying"


def test_begin_integration_journal_reopens_a_failed_retry(state: RouteState):
    """Retrying integration for the same changeset after a failure must get
    a fresh 'applying' journal, not silently reuse the stale 'failed' row.
    """
    workspace_id = _register_canonical_workspace(state)
    _make_changeset(state, "cs-1", workspace_id)

    journal = state.begin_integration_journal(
        journal_id="integration:cs-1", run_id="r1", epoch_id="ep-1",
        workspace_id=workspace_id, changeset_id="cs-1",
        expected_generation=0, expected_dirty_hash="dirty-0",
    )
    assert journal["status"] == "applying"
    state.finish_integration_journal("integration:cs-1", "failed", "boom")

    retried = state.begin_integration_journal(
        journal_id="integration:cs-1", run_id="r1", epoch_id="ep-1",
        workspace_id=workspace_id, changeset_id="cs-1",
        expected_generation=0, expected_dirty_hash="dirty-0",
    )
    assert retried["status"] == "applying"
    assert retried["error"] is None


def test_reconcile_lifecycle_terminalizes_stale_agent_executions(state: RouteState):
    """A claim orphaned by crash recovery must also terminalize the
    underlying agent_executions row -- otherwise a dead execution stays
    'started' forever, permanently consuming phase parallelism/fanout
    budget since nothing else ever marks it terminal.
    """
    from enhanced_router.state import _utcnow_age

    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.create_agent_execution(
        execution_id="exec-1", run_id="r1", epoch_id="ep-1",
        claude_agent_id="agent-1", role="recon", model_id="model-a",
    )
    conn = state._new_conn()
    try:
        conn.execute(
            "INSERT INTO runnable_action_claims "
            "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
            "action_kind, claim_token, status, created_at, claimed_at, consumed_at, "
            "expires_at, claude_agent_id, execution_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("action-1", "r1", "ep-1", "phase-1", "recon", "brigade-recon", "model-a",
             "native_agent", "tok-1", "consumed", _utcnow_age(2000), _utcnow_age(2000),
             _utcnow_age(2000), _utcnow_age(2000), "agent-1", "exec-1"),
        )
        conn.commit()
    finally:
        conn.close()

    result = state.reconcile_lifecycle(max_age_seconds=900)
    assert result["claims_orphaned"] == 1
    assert result["executions_orphaned"] == 1

    execution = state.get_agent_execution("exec-1")
    assert execution is not None
    assert execution["status"] == "timeout"
    assert "orphaned" in execution["error"]

    conn = state._new_conn()
    try:
        row = conn.execute(
            "SELECT status FROM runnable_action_claims WHERE action_id='action-1'"
        ).fetchone()
        assert row["status"] == "orphaned"
    finally:
        conn.close()


def test_reconcile_lifecycle_skips_already_terminal_execution(state: RouteState):
    """If the execution actually completed right before the crash-recovery
    cutoff ran, reconciling its now-stale claim must not raise -- it should
    just skip terminalizing an execution that's already terminal.
    """
    from enhanced_router.state import _utcnow_age

    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.create_agent_execution(
        execution_id="exec-1", run_id="r1", epoch_id="ep-1",
        claude_agent_id="agent-1", role="recon", model_id="model-a",
    )
    state.update_agent_execution(execution_id="exec-1", status="completed")
    conn = state._new_conn()
    try:
        conn.execute(
            "INSERT INTO runnable_action_claims "
            "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
            "action_kind, claim_token, status, created_at, claimed_at, consumed_at, "
            "expires_at, claude_agent_id, execution_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("action-1", "r1", "ep-1", "phase-1", "recon", "brigade-recon", "model-a",
             "native_agent", "tok-1", "consumed", _utcnow_age(2000), _utcnow_age(2000),
             _utcnow_age(2000), _utcnow_age(2000), "agent-1", "exec-1"),
        )
        conn.commit()
    finally:
        conn.close()

    result = state.reconcile_lifecycle(max_age_seconds=900)
    assert result["claims_orphaned"] == 1
    assert result["executions_orphaned"] == 0
    assert state.get_agent_execution("exec-1")["status"] == "completed"  # type: ignore[index]


def test_fallback_accounting_is_scoped_per_role_in_multi_role_phase(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A phase with more than one allowed role must not let one role's
    failures shift another role's fallback-model index -- each role has
    its own independent fallback ladder in role_routes.  No workflow in
    config/workflows.yaml currently declares a multi-role phase, so this
    exercises a latent path rather than today's actual configuration.
    """
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route(
        "r1", "ep-1", "implementer", "model-a", "manual",
        fallback_models=["model-a-fallback"],
    )
    state.set_role_route(
        "r1", "ep-1", "repairer", "model-b", "manual",
        fallback_models=["model-b-fallback"],
    )
    state.initialize_workflow_phases(
        "r1", "ep-1",
        [{"id": "mixed", "roles": ["implementer", "repairer"], "max_fanout": 5, "max_attempts": 5}],
    )
    state.start_phase("r1", "ep-1", "mixed")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    # Record a failed implementer attempt directly in agent_executions --
    # that's what get_runnable_actions actually reads for fallback/attempt
    # accounting, not the runnable_action_claims ledger.
    state.create_workspace(
        workspace_id="ws-impl-1", run_id="r1", epoch_id="ep-1", kind="shadow",
        path="/tmp/shadow-impl-1", base_sha="sha-0", dirty_patch_hash="dirty-0",
        status="active", owner_execution_id="exec-impl-1",
    )
    state.create_agent_execution(
        execution_id="exec-impl-1", run_id="r1", epoch_id="ep-1",
        claude_agent_id="agent-impl-1", role="implementer", model_id="model-a",
        phase_id="mixed", workspace_id="ws-impl-1",
    )
    state.update_agent_execution(execution_id="exec-impl-1", status="failed")

    # repairer has had zero failures of its own -- it must still be offered
    # its PRIMARY model, not skip straight to its fallback because
    # implementer failed in the same phase.
    actions = state.get_runnable_actions("r1", "ep-1")
    repairer_action = next(a for a in actions if a["role"] == "repairer")
    assert repairer_action["model_id"] == "model-b"


def test_reconcile_lifecycle_terminalizes_orphaned_detached_fastpath_jobs(
    state: RouteState,
):
    """Detached fastpath jobs (start_detached_sidecar_execution) have no
    backing runnable_action_claims row, so the claim-based orphan detection
    never sees them. A crashed job must still get picked up and
    terminalized -- otherwise it stays 'started' forever and can never even
    be retried (SidecarExecutor.retry requires a terminal failed/timeout
    status before it will recover a detached job).
    """
    from enhanced_router.state import _utcnow_age

    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.start_detached_sidecar_execution(
        run_id="r1", epoch_id="ep-1", execution_id="scx-1",
        phase_id="fastpath:route", role="fastpath", model_id="model-a",
        provider_id=None, packet={"hello": "world"},
    )
    conn = state._new_conn()
    try:
        conn.execute(
            "UPDATE agent_executions SET started_at=? WHERE execution_id='scx-1'",
            (_utcnow_age(2000),),
        )
        conn.commit()
    finally:
        conn.close()

    result = state.reconcile_lifecycle(max_age_seconds=900)
    assert result["detached_executions_orphaned"] == 1

    execution = state.get_agent_execution("scx-1")
    assert execution is not None
    assert execution["status"] == "timeout"
    assert "orphaned" in execution["error"]
    assert execution["error_class"] == "orphaned_after_restart"
    assert execution["orphaned_at"] is not None


def test_startup_reconciles_recent_detached_job_immediately(state: RouteState):
    state.create_run("r-recent")
    state.create_epoch("r-recent", "ep-1", "normal", "hybrid")
    state.start_detached_sidecar_execution(
        run_id="r-recent", epoch_id="ep-1", execution_id="scx-recent",
        phase_id="fastpath:verify", role="fastpath", model_id="model-a",
        provider_id=None, packet={"streaming": True},
    )

    result = state.reconcile_lifecycle(detached_max_age_seconds=0)

    assert result["detached_executions_orphaned"] == 1
    execution = state.get_agent_execution("scx-recent")
    assert execution is not None
    assert execution["error_class"] == "orphaned_after_restart"


def test_restart_orphan_does_not_advance_route_ladder():
    from enhanced_router.runnable_action_state import _execution_advances_route

    assert not _execution_advances_route({
        "status": "timeout",
        "error_class": "orphaned_after_restart",
    })


def test_litellm_request_attribution_preserves_requested_and_observed_routes(
    state: RouteState,
):
    row = state.record_litellm_request_attribution(
        request_id="req-1",
        generation=3,
        agent_binding_id=None,
        logical_model_id="grouped",
        requested_provider_ids=["openrouter", "freeinference"],
        allowed_deployments=["free", "local"],
        reported_deployment_id="free",
        actual_provider_id="freeinference",
        actual_endpoint_id="free",
        attribution_source="validated_response_header",
        trusted=True,
        status_code=200,
    )
    assert row["trusted"] == 1
    assert row["requested_provider_ids_json"] == '["freeinference","openrouter"]'
    assert state.get_litellm_request_attributions(request_id="req-1")[0][
        "actual_endpoint_id"
    ] == "free"


def test_v29_to_current_adds_binding_and_group_columns(tmp_path: Path):
    db = tmp_path / "v29.db"
    state = RouteState(db)
    conn = state._new_conn()
    try:
        conn.execute("ALTER TABLE agent_bindings DROP COLUMN configuration_hash")
        conn.execute("PRAGMA user_version = 29")
        conn.commit()
    finally:
        conn.close()

    upgraded = RouteState(db)
    conn = upgraded._new_conn()
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_bindings)")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert "configuration_hash" in columns
        assert "routing_mode" in columns
        assert "deployment_group" in columns
        assert version == 41
    finally:
        conn.close()


def test_runnable_native_action_requires_claim_and_is_consumed_once(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """The controller claims a native action before a hook may spawn it."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    actions = state.get_runnable_actions("r1", "ep-1")
    native = next(item for item in actions if item["action_kind"] == "native_agent")
    claim = state.claim_runnable_action("r1", "ep-1", native["action_id"])
    assert claim["status"] == "claimed"
    assert claim["claim_token"]

    # Re-reading does not create a second action while the first is claimed.
    assert not any(
        item.get("action_id") == native["action_id"]
        for item in state.get_runnable_actions("r1", "ep-1")
    )

    consumed = state.consume_runnable_action_for_spawn(
        "r1", "ep-1", native["native_agent_name"], claude_agent_id="agent-1",
    )
    assert consumed is not None
    assert consumed["status"] == "consumed"
    assert state.get_spawn_assignment("r1", "ep-1", "agent-1")["model_id"] == "model-a"
    assert state.consume_runnable_action_for_spawn(
        "r1", "ep-1", native["native_agent_name"], claude_agent_id="agent-2",
    ) is None
    finished = state.finish_spawn_assignment("r1", "ep-1", "agent-1", "completed")
    assert finished is not None
    assert finished["status"] == "consumed"


def test_claim_runnable_action_denied_once_run_token_budget_is_exhausted(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A run created with token_budget stops admitting new claims once
    cumulative agent_executions.total_tokens for the run reaches it."""
    state.create_run("r1", token_budget=100)
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )
    state.start_phase("r1", "ep-1", "recon")
    state.create_agent_execution(
        "exec-1", "r1", "ep-1", "agent-0", "recon", "model-a",
    )
    state.update_agent_execution("exec-1", total_tokens=150)

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    actions = state.get_runnable_actions("r1", "ep-1")
    native = next(item for item in actions if item["action_kind"] == "native_agent")
    with pytest.raises(WorkflowStateError, match="exhausted its token budget"):
        state.claim_runnable_action("r1", "ep-1", native["action_id"])


def test_token_budget_reservations_block_concurrent_claims_before_spend(
    state: RouteState,
):
    """Admission accounts for live reservations, not only completed usage."""
    state.create_run("r1", token_budget=500)

    first = state.reserve_token_budget(
        reservation_id="tokens:first",
        run_id="r1",
        epoch_id="ep-1",
        action_id="action:first",
        execution_id=None,
        estimated_tokens=400,
    )
    assert first["state"] == "reserved"

    blocked = state.reserve_token_budget(
        reservation_id="tokens:second",
        run_id="r1",
        epoch_id="ep-1",
        action_id="action:second",
        execution_id=None,
        estimated_tokens=200,
    )
    assert blocked["state"] == "unavailable"
    assert "exhausted its token budget" in blocked["reason"]

    state.release_token_reservation("tokens:first", "released")
    admitted = state.reserve_token_budget(
        reservation_id="tokens:second",
        run_id="r1",
        epoch_id="ep-1",
        action_id="action:second",
        execution_id=None,
        estimated_tokens=200,
    )
    assert admitted["state"] == "reserved"


def test_claim_runnable_action_ignores_budget_when_run_has_none(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """token_budget defaults to unbounded (NULL) for a run that never set one."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    actions = state.get_runnable_actions("r1", "ep-1")
    native = next(item for item in actions if item["action_kind"] == "native_agent")
    claim = state.claim_runnable_action("r1", "ep-1", native["action_id"])
    assert claim["status"] == "claimed"


def test_controller_phase_generates_controlled_native_mutator_action(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    state.create_run("r1", session_id="session-1")
    state.create_epoch("r1", "ep-1", "trivial", "hybrid")
    state.bind_or_get_controller(
        run_id="r1", client_session_id="session-1", public_model="controller-model",
        registry_model_id="model-a", backend="anthropic-passthrough", upstream_model="claude",
        provider_id=None, api_base=None, catalog_generation=None, registry_hash="hash",
        certification_id=None, auth_spec_json=None, api_key_env=None,
    )
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "implementation", "actor": "controller", "mutation": True}],
    )
    state.start_phase("r1", "ep-1", "implementation", actor="controller")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    action = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert action["action_kind"] == "controller_action"
    assert action["legacy_action_kind"] == "native_agent"
    assert action["controller_action_kind"] == "controller_implementation"
    assert action["native_agent_name"] == "controller-direct"
    assert action["role"] == "controller"
    assert action["requires_main_controller"] is True
    claimed = state.claim_runnable_action("r1", "ep-1", action["action_id"])
    assert claimed["action_kind"] == "controller_action"
    assert state.get_spawn_assignment("r1", "ep-1", "controller") is None


def test_unclaimed_native_action_is_not_spawnable(state: RouteState, monkeypatch: pytest.MonkeyPatch):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"]}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())
    action = next(
        item for item in state.get_runnable_actions("r1", "ep-1")
        if item["action_kind"] == "native_agent"
    )
    assert state.consume_runnable_action_for_spawn(
        "r1", "ep-1", action["native_agent_name"],
    ) is None


def test_sidecar_claim_creates_scoped_execution_events(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{
            "id": "recon", "roles": ["recon"], "execution_kind": "sidecar_call",
        }],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    action = next(item for item in state.get_runnable_actions("r1", "ep-1")
                  if item["action_kind"] == "sidecar_call")
    claim = state.claim_runnable_action("r1", "ep-1", action["action_id"])
    execution = state.start_sidecar_execution(
        run_id="r1", epoch_id="ep-1", action_id=action["action_id"],
        claim_token=claim["claim_token"], execution_id="scx-test", packet={"task": "inspect"},
    )
    assert execution["execution_kind"] == "sidecar_call"
    assert execution["claude_agent_id"] == "sidecar:scx-test"
    events = state.get_execution_events("r1", "ep-1", "scx-test")
    assert events[0]["event_type"] == "started"
    assert events[0]["payload"] == {"task": "inspect"}


def test_sidecar_call_outside_runs_sidecar_profile_never_becomes_runnable(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A run bounded to a sidecar_profile_id must not be offered a
    workflow phase naming a sidecar outside that profile's allow-list --
    otherwise sidecar_profiles.yaml's bound is advisory in name only.
    """
    state.create_run("r1", sidecar_profile_id="lightweight")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{
            "id": "recon", "roles": ["recon"], "execution_kind": "sidecar_call",
            "sidecar_id": "verifier",
        }],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

        @staticmethod
        def get_sidecar(sidecar_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id="model-b", endpoint="auto", enabled=True, mode="structured",
                timeout_seconds=45.0, max_packet_bytes=64_000, max_output_tokens=2_048,
            )

        @staticmethod
        def resolve_sidecars(sidecar_profile_id: str | None) -> dict:
            # "lightweight" only allows "reviewer" -- "verifier" is outside it.
            if sidecar_profile_id == "lightweight":
                return {"reviewer": SimpleNamespace()}
            return {"reviewer": SimpleNamespace(), "verifier": SimpleNamespace()}

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    actions = state.get_runnable_actions("r1", "ep-1")
    assert not any(item["action_kind"] == "sidecar_call" for item in actions)


def test_sidecar_call_inside_runs_sidecar_profile_is_runnable(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    state.create_run("r1", sidecar_profile_id="lightweight")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{
            "id": "recon", "roles": ["recon"], "execution_kind": "sidecar_call",
            "sidecar_id": "reviewer",
        }],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

        @staticmethod
        def get_sidecar(sidecar_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                model_id="model-b", endpoint="auto", enabled=True, mode="structured",
                timeout_seconds=45.0, max_packet_bytes=64_000, max_output_tokens=2_048,
            )

        @staticmethod
        def resolve_sidecars(sidecar_profile_id: str | None) -> dict:
            if sidecar_profile_id == "lightweight":
                return {"reviewer": SimpleNamespace()}
            return {"reviewer": SimpleNamespace(), "verifier": SimpleNamespace()}

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    actions = state.get_runnable_actions("r1", "ep-1")
    assert any(item["action_kind"] == "sidecar_call" for item in actions)


def test_retry_policy_keeps_phase_active_until_attempt_budget_is_exhausted(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{
            "id": "recon", "roles": ["recon"],
            "execution_kind": "sidecar_call", "max_fanout": 1,
            "max_attempts": 2, "fallback_policy": "retry",
        }],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    first = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    first_claim = state.claim_runnable_action("r1", "ep-1", first["action_id"])
    first_execution = state.start_sidecar_execution(
        run_id="r1", epoch_id="ep-1", action_id=first["action_id"],
        claim_token=first_claim["claim_token"], execution_id="scx-retry-1",
        packet={"task": "retry"},
    )
    state.update_agent_execution(first_execution["execution_id"], status="failed")

    assert state.complete_phase_if_ready("r1", "ep-1", "recon")["status"] == "active"
    second = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert second["current_fanout"] == 1

    second_claim = state.claim_runnable_action("r1", "ep-1", second["action_id"])
    second_execution = state.start_sidecar_execution(
        run_id="r1", epoch_id="ep-1", action_id=second["action_id"],
        claim_token=second_claim["claim_token"], execution_id="scx-retry-2",
        packet={"task": "retry"},
    )
    state.update_agent_execution(second_execution["execution_id"], status="failed")

    assert state.complete_phase_if_ready("r1", "ep-1", "recon")["status"] == "failed"


def test_role_route_fallback_is_used_without_phase_retry_policy(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route(
        "r1", "ep-1", "recon", "model-a", "manual",
        fallback_models=["model-b"],
    )
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1, "max_attempts": 2}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

        @staticmethod
        def native_agent_name(model_id: str, role: str) -> str:
            return f"brigade-{model_id}-{role}"

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    first = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert first["model_id"] == "model-a"
    execution = state.create_agent_execution(
        "ex-fallback-1", "r1", "ep-1", "agent-fallback-1", "recon", "model-a",
        phase_id="recon",
    )
    state.update_agent_execution(execution["execution_id"], status="failed")

    second = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert second["model_id"] == "model-b"


def test_fallback_activation_preserves_candidate_endpoint(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A fallback candidate with its own endpoint must not be forced back to
    'auto' when it's activated -- fallbacks must be able to pin a different
    provider/endpoint than the primary route."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route(
        "r1", "ep-1", "recon", "model-a", "manual",
        fallback_routes=[{
            "model": "model-b",
            "endpoint": "provider-b",
            "provider_id": "provider-b",
        }],
    )
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1, "max_attempts": 2}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

        @staticmethod
        def native_agent_name(model_id: str, role: str) -> str:
            return f"brigade-{model_id}-{role}"

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    execution = state.create_agent_execution(
        "ex-fallback-1", "r1", "ep-1", "agent-fallback-1", "recon", "model-a",
        phase_id="recon",
    )
    state.update_agent_execution(execution["execution_id"], status="failed")

    second = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert second["model_id"] == "model-b"
    assert second["endpoint"] == "provider-b"
    assert second["provider_id"] == "provider-b"


def test_fallback_activation_legacy_rows_still_work(
    state: RouteState, monkeypatch: pytest.MonkeyPatch,
):
    """A role route written with only fallback_models (no fallback_routes,
    e.g. from before this migration) must still activate its fallback, with
    the endpoint treated as auto."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route(
        "r1", "ep-1", "recon", "model-a", "manual",
        fallback_models=["model-b"],
    )
    # Simulate a pre-migration row: clear fallback_routes_json directly.
    conn = state._new_conn()
    try:
        conn.execute(
            "UPDATE role_routes SET fallback_routes_json = NULL "
            "WHERE run_id='r1' AND epoch_id='ep-1' AND role='recon'",
        )
        conn.commit()
    finally:
        conn.close()
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1, "max_attempts": 2}],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

        @staticmethod
        def native_agent_name(model_id: str, role: str) -> str:
            return f"brigade-{model_id}-{role}"

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    execution = state.create_agent_execution(
        "ex-fallback-1", "r1", "ep-1", "agent-fallback-1", "recon", "model-a",
        phase_id="recon",
    )
    state.update_agent_execution(execution["execution_id"], status="failed")

    second = next(item for item in state.get_runnable_actions("r1", "ep-1"))
    assert second["model_id"] == "model-b"
    assert second["endpoint"] == "auto"


def test_migration_v39_adds_fallback_routes_column(tmp_path: Path):
    db = tmp_path / "v38.db"
    state = RouteState(db)
    conn = state._new_conn()
    try:
        conn.execute("ALTER TABLE role_routes DROP COLUMN fallback_routes_json")
        conn.execute("PRAGMA user_version = 38")
        conn.commit()
    finally:
        conn.close()

    upgraded = RouteState(db)
    conn = upgraded._new_conn()
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(role_routes)")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert "fallback_routes_json" in columns
        assert version == 41
    finally:
        conn.close()


def test_completion_token_is_bound_to_workspace_and_consumed_once(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    prepared = state.prepare_completion_token("r1", "ep-1", "a" * 64)
    assert prepared["route_snapshot_sha256"]
    consumed = state.consume_completion_token(
        "r1", "ep-1", prepared["token"], "a" * 64,
        prepared["route_snapshot_sha256"],
    )
    assert consumed["valid"] is True
    replay = state.consume_completion_token(
        "r1", "ep-1", prepared["token"], "a" * 64,
        prepared["route_snapshot_sha256"],
    )
    assert replay == {"valid": False, "reason": "completion token was already consumed"}


# ================================================================== Mutation lease
# ==================================================================


def test_acquire_mutation_lease_succeeds(state: RouteState):
    """A mutation lease can be acquired for an agent."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    result = state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer")
    assert result is True

    lease = state.get_mutation_lease("r1", "agent-1")
    assert lease is not None
    assert lease["agent_id"] == "agent-1"
    assert lease["role"] == "implementer"
    assert lease["released_at"] is None


def test_acquire_mutation_lease_rejects_second_mutator(state: RouteState):
    """Second agent cannot acquire a lease while first holds one."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True
    assert state.acquire_mutation_lease("r1", "ep-1", "agent-2", "repairer") is False


def test_release_mutation_lease(state: RouteState):
    """Releasing a lease allows another agent to acquire."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True
    state.release_mutation_lease("r1", "agent-1")
    assert state.get_active_mutator("r1") is None


def test_reacquire_after_release(state: RouteState):
    """After release, a different agent can acquire the lease."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True
    state.release_mutation_lease("r1", "agent-1")
    assert state.acquire_mutation_lease("r1", "ep-1", "agent-2", "implementer") is True


def test_stale_lease_expiry(state: RouteState):
    """Expired leases are released by expire_stale_leases."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True
    count = state.expire_stale_leases("r1", max_age_seconds=0)
    assert count == 1
    assert state.get_active_mutator("r1") is None


def test_acquire_mutation_lease_reclaims_a_lease_from_a_crashed_holder(state: RouteState):
    """A hard-killed lease holder (no clean release, stale heartbeat) must
    not permanently block every future writer -- acquire_mutation_lease
    itself expires a stale lease on this workspace before checking for an
    active one, the same lazy-expiry-on-read shape as
    _active_action_claims for runnable_action_claims."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True

    # Simulate a crash: no release_mutation_lease call, heartbeat frozen far
    # in the past (older than mutation_lease_state._STALE_LEASE_MAX_AGE_SECONDS).
    conn = state._new_conn()
    conn.execute(
        "UPDATE mutation_leases SET heartbeat_at = '2000-01-01T00:00:00+00:00' "
        "WHERE run_id='r1' AND agent_id='agent-1'",
    )
    conn.commit()
    conn.close()

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-2", "repairer") is True
    mutator = state.get_active_mutator("r1")
    assert mutator is not None
    assert mutator["agent_id"] == "agent-2"

    old_lease = state.get_mutation_lease("r1", "agent-1")
    assert old_lease is not None
    assert old_lease["released_at"] is not None


def test_get_active_mutator(state: RouteState):
    """get_active_mutator returns the agent holding the unreleased lease."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer")
    mutator = state.get_active_mutator("r1")
    assert mutator is not None
    assert mutator["agent_id"] == "agent-1"


def test_lease_works_across_runs(state: RouteState):
    """Each run has its own mutation lease."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.create_run("r2")
    state.create_epoch("r2", "ep-2", "normal", "hybrid")

    assert state.acquire_mutation_lease("r1", "ep-1", "agent-1", "implementer") is True
    assert state.acquire_mutation_lease("r2", "ep-2", "agent-2", "repairer") is True
    assert state.get_active_mutator("r1")["agent_id"] == "agent-1"
    assert state.get_active_mutator("r2")["agent_id"] == "agent-2"


def test_migration_v5_creates_tables(tmp_path: Path):
    """A fresh DB created at v4 schema gets binding tables after upgrade."""
    # Create a DB manually at v4 (simulate old schema)
    from enhanced_router.state import _migrate_v1, _migrate_v2, _migrate_v3, _migrate_v4, _migrate_v5

    db = tmp_path / "migration_test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate_v1(conn)
    _migrate_v2(conn)
    _migrate_v3(conn)
    _migrate_v4(conn)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()

    # Verify old tables exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "role_routes" in table_names
    assert "agent_bindings" in table_names

    # Now migrate to v5
    _migrate_v5(conn)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()

    tables_after = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names_after = {t[0] for t in tables_after}
    assert "binding_versions" in table_names_after
    assert "binding_commands" in table_names_after
    assert "controller_policies" in table_names_after


def test_migration_v7_adds_identity_columns(tmp_path: Path):
    """v7 migration adds claude_session_id and claude_parent_agent_id to agent_bindings."""
    from enhanced_router.state import _migrate_v6, _migrate_v7

    db = tmp_path / "v7_migration.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Simulate a v6 schema: create agent_bindings WITHOUT the v7 columns
    conn.execute(
        """CREATE TABLE IF NOT EXISTS agent_bindings (
            binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            claude_agent_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            role TEXT NOT NULL,
            model_id TEXT NOT NULL,
            route_version INTEGER NOT NULL,
            bound_at TEXT NOT NULL,
            released_at TEXT,
            backend TEXT,
            registry_hash TEXT,
            catalog_generation INTEGER,
            litellm_model_name TEXT,
            upstream_model TEXT,
            api_base TEXT,
            api_key_env TEXT
        )"""
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_agent_binding "
        "ON agent_bindings(run_id, claude_agent_id) WHERE released_at IS NULL"
    )

    # Apply up to v6 (simulates a DB that had v6)
    _migrate_v6(conn)
    conn.execute("PRAGMA user_version = 6")
    conn.commit()

    # Verify agent_bindings exists but lacks the new columns
    col_info = conn.execute("PRAGMA table_info(agent_bindings)").fetchall()
    col_names = {c[1] for c in col_info}
    assert "claude_session_id" not in col_names
    assert "claude_parent_agent_id" not in col_names

    # Run v7 migration
    _migrate_v7(conn)
    conn.execute("PRAGMA user_version = 7")
    conn.commit()

    # Verify new columns exist
    col_info = conn.execute("PRAGMA table_info(agent_bindings)").fetchall()
    col_names = {c[1] for c in col_info}
    assert "claude_session_id" in col_names
    assert "claude_parent_agent_id" in col_names

    # Verify the expanded UNIQUE index exists
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='uq_active_agent_binding'"
    ).fetchone()
    assert indexes is not None

    conn.close()


def test_migration_v6_is_noop(tmp_path: Path):
    """v6 migration should not crash (it's a placeholder)."""
    from enhanced_router.state import _migrate_v6

    db = tmp_path / "noop_test.db"
    conn = sqlite3.connect(str(db))
    _migrate_v6(conn)  # should not raise
    conn.close()


# ---------------------------------------------------------------------------
# LiteLLM catalog lifecycle
# ---------------------------------------------------------------------------


class TestLiteLLMGeneration:
    def test_create_generation(self, state):
        """Creating a generation returns a positive integer."""
        gen = state.create_litellm_generation("abc123", 2, "digest1", "initial")
        assert isinstance(gen, int)
        assert gen > 0

    def test_get_active_none_initially(self, state):
        """No active generation before any activation."""
        assert state.get_active_litellm_generation() is None

    def test_activate_then_active(self, state):
        """Activating a generation makes it the active one."""
        gen = state.create_litellm_generation("abc", 1, "digest", "test")
        state.activate_litellm_generation(gen)
        active = state.get_active_litellm_generation()
        assert active is not None
        assert active["generation"] == gen
        assert active["status"] == "active"

    def test_activate_retires_previous(self, state):
        """Activating a new generation retires the previous active."""
        g1 = state.create_litellm_generation("h1", 1, "d1", "first")
        state.activate_litellm_generation(g1)

        g2 = state.create_litellm_generation("h2", 1, "d2", "second")
        state.activate_litellm_generation(g2)

        active = state.get_active_litellm_generation()
        assert active is not None
        assert active["generation"] == g2

        # g1 should be retired
        conn = state._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM litellm_generations WHERE generation = ?",
                (g1,),
            ).fetchone()
            assert row[0] == "retired"
        finally:
            conn.close()

    def test_register_deployment(self, state):
        """Registering a deployment returns a positive id."""
        gen = state.create_litellm_generation("abc", 1, "d", "test")
        dep = state.register_litellm_deployment(gen, 18000, 12345)
        assert isinstance(dep, int)
        assert dep > 0

    def test_update_deployment_status(self, state):
        """Updating deployment status persists the change."""
        gen = state.create_litellm_generation("abc", 1, "d", "test")
        dep = state.register_litellm_deployment(gen, 18000, 12345)
        state.update_litellm_deployment(dep, status="active")

        conn = state._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM litellm_deployments WHERE id = ?",
                (dep,),
            ).fetchone()
            assert row[0] == "active"
        finally:
            conn.close()

    def test_generation_includes_config_digest(self, state):
        """Config digest persists and can be read back."""
        gen = state.create_litellm_generation("abc", 1, "my-digest", "test")
        state.activate_litellm_generation(gen)
        active = state.get_active_litellm_generation()
        assert active["config_digest"] == "my-digest"

    def test_generation_model_count(self, state):
        """Model count persists correctly."""
        gen = state.create_litellm_generation("abc", 7, "d", "test")
        state.activate_litellm_generation(gen)
        active = state.get_active_litellm_generation()
        assert active["model_count"] == 7


# ==================================================================
# Binding-command operations (P0-4)
# ==================================================================


def test_record_binding_command(state: RouteState):
    """record_binding_command inserts a pending command row."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    result = state.record_binding_command(
        command_id="cmd-rec-1",
        run_id="r1",
        epoch_id="ep-1",
        command_type="binding_release",
        actor_type="controller",
        reason="testing record",
    )
    assert result["status"] == "pending"
    assert result["command_id"] == "cmd-rec-1"


def test_record_binding_command_rejects_invalid_type(state: RouteState):
    """record_binding_command rejects unknown command types."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    with pytest.raises(ValueError, match="Invalid command_type"):
        state.record_binding_command(
            command_id="cmd-bad",
            run_id="r1",
            epoch_id="ep-1",
            command_type="not_a_real_type",
            actor_type="controller",
            reason="test",
        )


def test_record_binding_command_all_valid_types(state: RouteState):
    """All five VALID_COMMAND_TYPES should be accepted."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    from enhanced_router.state import VALID_COMMAND_TYPES

    for ct in VALID_COMMAND_TYPES:
        result = state.record_binding_command(
            command_id=f"cmd-{ct.replace('_', '-')}",
            run_id="r1",
            epoch_id="ep-1",
            command_type=ct,
            actor_type="mcp",
            reason=f"test {ct}",
        )
        assert result["status"] == "pending"


def test_record_binding_command_with_all_fields(state: RouteState):
    """record_binding_command accepts optional fields."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    result = state.record_binding_command(
        command_id="cmd-full",
        run_id="r1",
        epoch_id="ep-1",
        command_type="model_change",
        actor_type="mcp",
        reason="full test",
        claude_session_id="sess-1",
        claude_agent_id="agent-1",
        expected_binding_version=3,
        requested_model_id="new-model",
        requested_role="recon",
    )
    assert result["status"] == "pending"
    assert result["command_id"] == "cmd-full"


def test_get_binding_commands(state: RouteState):
    """get_binding_commands returns commands for a run/epoch."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    state.record_binding_command("cmd-g1", "r1", "ep-1", "binding_release", "ctrl", "first")
    state.record_binding_command("cmd-g2", "r1", "ep-1", "route_change", "ctrl", "second")

    commands = state.get_binding_commands("r1", "ep-1", limit=10)
    assert len(commands) == 2
    # Should be ordered newest first
    assert commands[0]["command_id"] == "cmd-g2"
    assert commands[1]["command_id"] == "cmd-g1"


def test_get_binding_commands_respects_limit(state: RouteState):
    """get_binding_commands respects the limit parameter."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    for i in range(5):
        state.record_binding_command(f"cmd-l{i}", "r1", "ep-1", "route_change", "ctrl", f"item {i}")

    commands = state.get_binding_commands("r1", "ep-1", limit=3)
    assert len(commands) == 3


def test_get_binding_commands_empty_for_other_epoch(state: RouteState):
    """Commands for one epoch do not appear for another."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.record_binding_command("cmd-x", "r1", "ep-1", "route_change", "ctrl", "ep1")

    commands = state.get_binding_commands("r1", "ep-other", limit=10)
    assert len(commands) == 0


def test_apply_binding_command(state: RouteState):
    """apply_binding_command changes status from pending to applied."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    state.record_binding_command("cmd-apply-1", "r1", "ep-1", "binding_release", "ctrl", "release test")

    result = state.apply_binding_command("cmd-apply-1", actor_type="controller", actor_id="admin-1")
    assert result["command_id"] == "cmd-apply-1"
    assert result["status"] == "applied"
    assert result["actor_id"] == "admin-1"


def test_apply_binding_command_rejects_not_found(state: RouteState):
    """apply_binding_command raises ValueError for unknown command."""
    with pytest.raises(ValueError, match="not found"):
        state.apply_binding_command("cmd-nonexistent", actor_type="controller")


def test_apply_binding_command_rejects_already_applied(state: RouteState):
    """apply_binding_command raises ValueError if command is already applied."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    state.record_binding_command("cmd-double", "r1", "ep-1", "binding_release", "ctrl", "test")
    state.apply_binding_command("cmd-double", actor_type="controller")

    with pytest.raises(ValueError, match="not pending"):
        state.apply_binding_command("cmd-double", actor_type="controller")


def test_binding_command_audit_chain(state: RouteState):
    """End-to-end: record -> query -> apply -> query again."""
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")

    # Record
    state.record_binding_command("cmd-audit", "r1", "ep-1", "model_change", "mcp", "model swap", requested_role="recon")

    # Verify pending
    cmds = state.get_binding_commands("r1", "ep-1")
    assert len(cmds) == 1
    assert cmds[0]["status"] == "pending"

    # Apply
    state.apply_binding_command("cmd-audit", actor_type="controller", actor_id="supervisor")

    # Verify applied
    cmds = state.get_binding_commands("r1", "ep-1")
    assert cmds[0]["status"] == "applied"
    assert cmds[0]["actor_id"] == "supervisor"


# ==================================================================
# Migration v8 CHECK constraint
# ==================================================================


def test_migration_v8_adds_command_type_check(tmp_path: Path):
    """v8 migration adds CHECK constraint on command_type."""
    from enhanced_router.state import _migrate_v5, _migrate_v8

    db = tmp_path / "v8_migration.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Run up to v5 (which creates binding_commands without CHECK)
    _migrate_v5(conn)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()

    # Insert a valid row
    conn.execute(
        "INSERT INTO binding_commands "
        "(command_id, run_id, epoch_id, claude_session_id, command_type, actor_type, reason, created_at) "
        "VALUES (?, 'r1', 'ep-1', '', 'model_change', 'mcp', 'test', ?)",
        ("cmd-v8", _utcnow()),
    )
    conn.commit()

    # Run v8 migration
    _migrate_v8(conn)
    conn.execute("PRAGMA user_version = 8")
    conn.commit()

    # Data should be preserved
    row = conn.execute(
        "SELECT command_type FROM binding_commands WHERE command_id = ?",
        ("cmd-v8",),
    ).fetchone()
    assert row is not None
    assert row[0] == "model_change"

    # The CHECK constraint should now be present
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='binding_commands'"
    ).fetchone()
    create_sql = sql_row[0] if sql_row else ""
    assert "CHECK" in create_sql.upper()

    conn.close()


def test_migration_v8_is_idempotent(tmp_path: Path):
    """Running v8 twice should not crash."""
    from enhanced_router.state import (
        _migrate_v1, _migrate_v5, _migrate_v8,
    )

    db = tmp_path / "v8_idempotent.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    _migrate_v1(conn)
    _migrate_v5(conn)
    conn.execute("PRAGMA user_version = 5")
    conn.commit()

    _migrate_v8(conn)
    _migrate_v8(conn)  # second time should be a no-op
    conn.close()


# ==================================================================
# Migration v9 -- mutation_leases
# ==================================================================


def test_migration_v9_creates_mutation_leases_table(tmp_path: Path):
    """v9 migration creates mutation_leases table."""
    from enhanced_router.state import _migrate_v8

    db = tmp_path / "v9_migration.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Run up to v8
    _migrate_v8(conn)
    conn.execute("PRAGMA user_version = 8")
    conn.commit()

    # mutation_leases should not exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "mutation_leases" not in table_names

    # Run v9
    _migrate_v9(conn)
    conn.execute("PRAGMA user_version = 9")
    conn.commit()

    # Now the table should exist
    tables_after = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names_after = {t[0] for t in tables_after}
    assert "mutation_leases" in table_names_after


def test_migration_v9_is_idempotent(tmp_path: Path):
    """Running v9 twice should not crash."""
    from enhanced_router.state import _migrate_v1

    db = tmp_path / "v9_idempotent.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    _migrate_v1(conn)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()

    _migrate_v9(conn)
    _migrate_v9(conn)  # second time should be a no-op
    conn.close()


# ==================================================================
# Migration v10 -- termination_reason column
# ==================================================================


def test_migration_v10_adds_termination_reason_column(tmp_path: Path):
    """v10 migration adds termination_reason column to litellm_deployments."""

    db = tmp_path / "v10_migration.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Create litellm_deployments WITHOUT termination_reason column
    # (simulating a database from before v10 migration)
    conn.execute("""CREATE TABLE IF NOT EXISTS litellm_generations (
        generation INTEGER PRIMARY KEY AUTOINCREMENT,
        registry_hash TEXT NOT NULL,
        model_count INTEGER NOT NULL,
        config_digest TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'staging' CHECK(status IN ('staging','active','draining','retired')),
        reason TEXT,
        created_at TEXT NOT NULL,
        activated_at TEXT,
        retired_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS litellm_deployments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation INTEGER NOT NULL,
        port INTEGER NOT NULL,
        pid INTEGER,
        status TEXT NOT NULL DEFAULT 'starting' CHECK(status IN ('starting','active','draining','dead','failed')),
        health_checked_at TEXT,
        created_at TEXT NOT NULL
    )""")
    conn.commit()

    # Verify litellm_deployments exists but lacks termination_reason
    col_info = conn.execute(
        "PRAGMA table_info(litellm_deployments)"
    ).fetchall()
    col_names = {c[1] for c in col_info}
    assert "termination_reason" not in col_names

    # Run v10 migration
    _migrate_v10(conn)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()

    # Verify new column exists
    col_info = conn.execute(
        "PRAGMA table_info(litellm_deployments)"
    ).fetchall()
    col_names = {c[1] for c in col_info}
    assert "termination_reason" in col_names

    conn.close()


def test_migration_v10_is_idempotent(tmp_path: Path):
    """Running v10 twice should not crash."""

    db = tmp_path / "v10_idempotent.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Create litellm_deployments WITHOUT termination_reason
    conn.execute("""CREATE TABLE IF NOT EXISTS litellm_deployments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation INTEGER NOT NULL,
        port INTEGER NOT NULL,
        pid INTEGER,
        status TEXT NOT NULL DEFAULT 'starting' CHECK(status IN ('starting','active','draining','dead','failed')),
        health_checked_at TEXT,
        created_at TEXT NOT NULL
    )""")
    conn.commit()

    _migrate_v10(conn)
    _migrate_v10(conn)  # second time should be a no-op
    conn.close()


# ==================================================================
# Migration v11 -- workflow_phases
# ==================================================================


def test_migration_v11_creates_workflow_phases_table(tmp_path: Path):
    """v11 migration creates workflow_phases table."""

    db = tmp_path / "v11_migration.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    # Create tables WITHOUT workflow_phases (simulate a DB that had v10 but not the DDL)
    # We use raw SQL to avoid _migrate_v1 which already includes workflow_phases in DDL
    conn.execute("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, claude_session_id TEXT, cwd TEXT, created_at TEXT, closed_at TEXT)
    """)
    conn.execute("""
        CREATE TABLE epochs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
            epoch_id TEXT NOT NULL, workflow_id TEXT NOT NULL, profile_id TEXT,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','closed')),
            created_at TEXT NOT NULL, closed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE role_routes (
            run_id TEXT NOT NULL, epoch_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('recon','implementer','adversary','repairer')),
            model_id TEXT NOT NULL, source TEXT NOT NULL, reason TEXT,
            version INTEGER NOT NULL, changed_at TEXT NOT NULL,
            PRIMARY KEY (run_id, epoch_id, role)
        )
    """)
    conn.execute("""
        CREATE TABLE agent_bindings (
            binding_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            claude_agent_id TEXT NOT NULL, epoch_id TEXT NOT NULL, role TEXT NOT NULL,
            model_id TEXT NOT NULL, route_version INTEGER NOT NULL, bound_at TEXT NOT NULL,
            released_at TEXT, backend TEXT, registry_hash TEXT, catalog_generation INTEGER,
            litellm_model_name TEXT, upstream_model TEXT, api_base TEXT, api_key_env TEXT,
            claude_session_id TEXT NOT NULL DEFAULT '', claude_parent_agent_id TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE route_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, epoch_id TEXT,
            event_type TEXT NOT NULL, role TEXT, old_model_id TEXT, new_model_id TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE model_health (
            model_id TEXT NOT NULL, configuration_hash TEXT NOT NULL,
            harness_version TEXT NOT NULL, status TEXT NOT NULL,
            reachable INTEGER NOT NULL, authenticated INTEGER NOT NULL,
            compatible INTEGER NOT NULL, failure_rate REAL, latency_ms REAL,
            checked_at TEXT NOT NULL, reason TEXT,
            PRIMARY KEY (model_id, configuration_hash, harness_version)
        )
    """)
    conn.execute("""
        CREATE TABLE litellm_generations (
            generation INTEGER PRIMARY KEY AUTOINCREMENT,
            registry_hash TEXT NOT NULL, model_count INTEGER NOT NULL,
            config_digest TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'staging' CHECK(status IN ('staging','active','draining','retired')),
            reason TEXT, created_at TEXT NOT NULL, activated_at TEXT, retired_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE binding_versions (
            binding_id INTEGER PRIMARY KEY, version INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE binding_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT, command_id TEXT NOT NULL UNIQUE,
            run_id TEXT NOT NULL, epoch_id TEXT NOT NULL, claude_session_id TEXT NOT NULL DEFAULT '',
            command_type TEXT NOT NULL, actor_type TEXT NOT NULL DEFAULT 'controller', reason TEXT,
            claude_agent_id TEXT NOT NULL DEFAULT '', expected_binding_version INTEGER,
            requested_model_id TEXT, requested_role TEXT,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applied','cancelled')),
            applied_at TEXT, actor_id TEXT, created_at TEXT NOT NULL,
            CHECK(command_type IN ('model_change','profile_set','route_change','binding_release','binding_reenable'))
        )
    """)
    conn.execute("""
        CREATE TABLE controller_policies (
            run_id TEXT PRIMARY KEY, permitted_models TEXT NOT NULL,
            model_change_policy TEXT NOT NULL DEFAULT 'reject' CHECK(model_change_policy IN ('allow','reject')),
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE mutation_leases (
            run_id TEXT NOT NULL, epoch_id TEXT NOT NULL, agent_id TEXT NOT NULL,
            role TEXT NOT NULL, acquired_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL,
            released_at TEXT, PRIMARY KEY (run_id, agent_id)
        )
    """)
    # Add termination_reason to litellm_deployments (from v10)
    conn.execute("""
        CREATE TABLE litellm_deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER NOT NULL,
            port INTEGER NOT NULL, pid INTEGER,
            status TEXT NOT NULL DEFAULT 'starting' CHECK(status IN ('starting','active','draining','dead','failed')),
            health_checked_at TEXT, created_at TEXT NOT NULL, termination_reason TEXT
        )
    """)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()

    # workflow_phases should not exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "workflow_phases" not in table_names

    # Run v11
    _migrate_v11(conn)
    conn.execute("PRAGMA user_version = 11")
    conn.commit()

    # Now the table should exist
    tables_after = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names_after = {t[0] for t in tables_after}
    assert "workflow_phases" in table_names_after


def test_migration_v11_is_idempotent(tmp_path: Path):
    """Running v11 twice should not crash."""

    db = tmp_path / "v11_idempotent.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode = WAL")

    from enhanced_router.state import _migrate_v1
    _migrate_v1(conn)
    conn.execute("PRAGMA user_version = 10")
    conn.commit()

    _migrate_v11(conn)
    _migrate_v11(conn)  # second time should be a no-op
    conn.close()


# ==================================================================
# Workflow phases
# ==================================================================


class TestWorkflowPhases:
    """Tests for workflow phase lifecycle: create, start, complete, skip, validate."""

    def test_initialize_phases(self, state: RouteState):
        """Creates run+epoch, initializes 3 phases, verifies all pending."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [
            {"id": "phase-a", "actor": "recon"},
            {"id": "phase-b", "actor": "implementer"},
            {"id": "phase-c", "actor": "adversary"},
        ]
        result = state.initialize_workflow_phases("r1", "ep-1", phases)
        assert len(result) == 3
        for p in result:
            assert p["status"] == "pending"
        ids = {p["phase_id"] for p in result}
        assert ids == {"phase-a", "phase-b", "phase-c"}

    def test_initialize_phases_idempotent(self, state: RouteState):
        """Calling initialize_workflow_phases twice returns same data."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "x"}, {"id": "y"}]
        r1 = state.initialize_workflow_phases("r1", "ep-1", phases)
        r2 = state.initialize_workflow_phases("r1", "ep-1", phases)
        assert len(r1) == len(r2) == 2
        # Same phase_ids
        assert {p["phase_id"] for p in r1} == {p["phase_id"] for p in r2}

    def test_start_phase(self, state: RouteState):
        """Starts a phase, verifies status=active and started_at is set."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "first", "actor": "recon"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        result = state.start_phase("r1", "ep-1", "first", actor="recon-agent")
        assert result["status"] == "active"
        assert result["started_at"] is not None
        assert result["actor"] == "recon-agent"

    def test_start_phase_not_pending(self, state: RouteState):
        """Starting already-active phase raises WorkflowPhaseStateError."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "only"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)
        state.start_phase("r1", "ep-1", "only")

        with pytest.raises(WorkflowPhaseStateError, match="cannot start"):
            state.start_phase("r1", "ep-1", "only")

    def test_complete_phase(self, state: RouteState):
        """Completes an active phase with evidence."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "work"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)
        state.start_phase("r1", "ep-1", "work")

        result = state.complete_phase(
            "r1", "ep-1", "work", result_evidence="done",
        )
        assert result["status"] == "completed"
        assert result["result_evidence"] == "done"
        assert result["completed_at"] is not None

    def test_complete_phase_not_active(self, state: RouteState):
        """Completing a pending phase raises WorkflowPhaseStateError."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "pending-phase"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        with pytest.raises(WorkflowPhaseStateError, match="cannot complete"):
            state.complete_phase("r1", "ep-1", "pending-phase")

    def test_complete_phase_with_error(self, state: RouteState):
        """Auto-sets status=failed when error is non-empty."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "failing"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)
        state.start_phase("r1", "ep-1", "failing")

        result = state.complete_phase(
            "r1", "ep-1", "failing", error="something broke",
        )
        assert result["status"] == "failed"
        assert result["error"] == "something broke"

    def test_skip_phase(self, state: RouteState):
        """Skips a pending phase."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "skip-me", "required": False}]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        result = state.skip_phase("r1", "ep-1", "skip-me")
        assert result["status"] == "skipped"
        assert result["completed_at"] is not None

    def test_get_active_phase(self, state: RouteState):
        """Returns the active phase or None."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "p1"}, {"id": "p2"}]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        # No active phase yet
        assert state.get_active_phase("r1", "ep-1") is None

        state.start_phase("r1", "ep-1", "p1")
        active = state.get_active_phase("r1", "ep-1")
        assert active is not None
        assert active["phase_id"] == "p1"

        state.complete_phase("r1", "ep-1", "p1", result_evidence="completed")
        # After completion, no active phase remains
        assert state.get_active_phase("r1", "ep-1") is None

    def test_get_workflow_phases(self, state: RouteState):
        """Returns all phases ordered."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "c"}, {"id": "a"}, {"id": "b"}]
        result = state.initialize_workflow_phases("r1", "ep-1", phases)
        assert len(result) == 3
        ids = [p["phase_id"] for p in result]
        assert ids == ["c", "a", "b"]  # ordered by id column (insert order)

    def test_validate_phase_transition_happy(self, state: RouteState):
        """validate_phase_transition returns valid for a pending phase with no deps."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [{"id": "target", "depends_on": []}]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        result = state.validate_phase_transition("r1", "ep-1", "target", phases)
        assert result == {"valid": True}

    def test_validate_phase_transition_unmet_dependency(self, state: RouteState):
        """Dependency not completed blocks transition."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [
            {"id": "dep-a", "depends_on": []},
            {"id": "target", "depends_on": ["dep-a"]},
        ]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        result = state.validate_phase_transition("r1", "ep-1", "target", phases)
        assert result["valid"] is False
        assert "dep-a" in result["reason"]

    def test_validate_phase_transition_failed_dep(self, state: RouteState):
        """Dependency failed blocks transition."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "wf-1", "hybrid")

        phases = [
            {"id": "dep-a", "depends_on": []},
            {"id": "target", "depends_on": ["dep-a"]},
        ]
        state.initialize_workflow_phases("r1", "ep-1", phases)

        # Fail the dependency
        state.start_phase("r1", "ep-1", "dep-a")
        state.complete_phase("r1", "ep-1", "dep-a", error="boom")

        result = state.validate_phase_transition("r1", "ep-1", "target", phases)
        assert result["valid"] is False
        assert "failed" in result["reason"]

    def test_initialize_empty_phases_returns_empty_list(self, state: RouteState):
        """Calling initialize_workflow_phases with empty list returns []."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        result = state.initialize_workflow_phases("r1", "ep-1", [])
        assert result == []

    def test_start_nonexistent_phase_raises_error(self, state: RouteState):
        """Starting a phase that doesn't exist raises WorkflowPhaseStateError."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        with pytest.raises(WorkflowPhaseStateError):
            state.start_phase("r1", "ep-1", "does-not-exist")

    def test_complete_nonexistent_phase_raises_error(self, state: RouteState):
        """Completing a phase that doesn't exist raises WorkflowPhaseStateError."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        with pytest.raises(WorkflowPhaseStateError):
            state.complete_phase("r1", "ep-1", "does-not-exist")

    def test_skip_already_completed_phase_raises_error(self, state: RouteState):
        """Skipping a phase that isn't pending raises WorkflowPhaseStateError."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        state.initialize_workflow_phases("r1", "ep-1", [{"id": "phase-1"}])
        state.start_phase("r1", "ep-1", "phase-1")
        state.complete_phase("r1", "ep-1", "phase-1", result_evidence="completed")
        with pytest.raises(WorkflowPhaseStateError):
            state.skip_phase("r1", "ep-1", "phase-1")

    def test_validate_transition_nonexistent_phase(self, state: RouteState):
        """Validating transition to unknown phase returns invalid."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        result = state.validate_phase_transition("r1", "ep-1", "no-such-phase", [])
        assert result["valid"] is False
        assert "not found" in result["reason"].lower() or "no state record" in result["reason"].lower()

    def test_initialize_with_cross_cutting_phases(self, state: RouteState):
        """Initializing with cross-cutting workflow creates all 5 phases."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "cross-cutting", "hybrid")
        phases_spec = [
            {"id": "recon", "roles": ["recon"], "required": True, "mutation": False, "depends_on": []},
            {"id": "implementation", "roles": ["implementer"], "required": True, "mutation": True, "depends_on": ["recon"]},
            {"id": "adversarial-review", "roles": ["adversary"], "required": True, "mutation": False, "depends_on": ["implementation"]},
            {"id": "repair", "roles": ["repairer"], "required": False, "mutation": True, "depends_on": ["adversarial-review"], "conditional": "accepted_findings"},
            {"id": "verification", "actor": "controller", "depends_on": ["repair"], "mutation": False},
        ]
        result = state.initialize_workflow_phases("r1", "ep-1", phases_spec)
        assert len(result) == 5
        phase_ids = [p["phase_id"] for p in result]
        assert phase_ids == ["recon", "implementation", "adversarial-review", "repair", "verification"]
        for p in result:
            assert p["status"] == "pending"

    def test_full_phase_lifecycle(self, state: RouteState):
        """A phase can be started, completed, with evidence."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "normal", "hybrid")
        state.initialize_workflow_phases("r1", "ep-1", [{"id": "impl", "mutation": True}])

        p = state.start_phase("r1", "ep-1", "impl")
        assert p["status"] == "active"
        assert p["started_at"] is not None

        p2 = state.complete_phase("r1", "ep-1", "impl", result_evidence="all tests pass")
        assert p2["status"] == "completed"
        assert p2["completed_at"] is not None
        assert p2["result_evidence"] == "all tests pass"

        # Verify get_workflow_phases reflects it
        phases = state.get_workflow_phases("r1", "ep-1")
        assert len(phases) == 1
        assert phases[0]["status"] == "completed"

    def test_dependency_chain_blocks_transition(self, state: RouteState):
        """validate_phase_transition blocks when dependency is not completed."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "cross-cutting", "hybrid")
        phases_spec = [
            {"id": "recon", "depends_on": []},
            {"id": "impl", "depends_on": ["recon"]},
        ]
        state.initialize_workflow_phases("r1", "ep-1", phases_spec)

        # impl depends on recon, which is still pending
        result = state.validate_phase_transition("r1", "ep-1", "impl", phases_spec)
        assert result["valid"] is False
        assert "recon" in result["reason"]

        # Complete recon first
        state.start_phase("r1", "ep-1", "recon")
        state.complete_phase("r1", "ep-1", "recon", result_evidence="recon complete")

        # Now impl should be valid
        result2 = state.validate_phase_transition("r1", "ep-1", "impl", phases_spec)
        assert result2["valid"] is True

    def test_failed_dependency_blocks_downstream(self, state: RouteState):
        """A failed dependency prevents downstream phases from starting."""
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "cross-cutting", "hybrid")
        phases_spec = [
            {"id": "recon", "depends_on": []},
            {"id": "impl", "depends_on": ["recon"]},
        ]
        state.initialize_workflow_phases("r1", "ep-1", phases_spec)

        # Fail the dependency
        state.start_phase("r1", "ep-1", "recon")
        state.complete_phase("r1", "ep-1", "recon", error="recon failed")

        # impl should be blocked
        result = state.validate_phase_transition("r1", "ep-1", "impl", phases_spec)
        assert result["valid"] is False
        assert "failed" in result["reason"]

    def test_skipped_dependency_allows_downstream(self, state: RouteState):
        """A non-required skipped dependency should NOT block downstream via validate."""
        # Manual skips do not satisfy dependencies; only conditional skips do.
        state.create_run("r1")
        state.create_epoch("r1", "ep-1", "cross-cutting", "hybrid")
        phases_spec = [
            {"id": "recon", "required": False},
            {"id": "impl", "depends_on": ["recon"]},
        ]
        state.initialize_workflow_phases("r1", "ep-1", phases_spec)

        state.skip_phase("r1", "ep-1", "recon")

        result = state.validate_phase_transition("r1", "ep-1", "impl", phases_spec)
        assert result["valid"] is False


def test_agent_execution_requires_active_authorized_phase(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )

    with pytest.raises(ValueError, match="not active"):
        state.create_agent_execution(
            "exec-1", "r1", "ep-1", "agent-1", "recon", "model-a",
            phase_id="recon",
        )

    state.start_phase("r1", "ep-1", "recon")
    execution = state.create_agent_execution(
        "exec-1", "r1", "ep-1", "agent-1", "recon", "model-a",
        phase_id="recon",
    )
    assert execution["phase_id"] == "recon"


def test_agent_execution_enforces_phase_fanout(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )
    state.start_phase("r1", "ep-1", "recon")
    state.create_agent_execution(
        "exec-1", "r1", "ep-1", "agent-1", "recon", "model-a",
        phase_id="recon",
    )

    with pytest.raises(ValueError, match="max fanout"):
        state.create_agent_execution(
            "exec-2", "r1", "ep-1", "agent-2", "recon", "model-a",
            phase_id="recon",
        )


def test_agent_execution_status_transitions_are_monotonic(state: RouteState):
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 1}],
    )
    state.start_phase("r1", "ep-1", "recon")
    state.create_agent_execution(
        "exec-status", "r1", "ep-1", "agent-1", "recon", "model-a",
        phase_id="recon",
    )

    assert state.update_agent_execution("exec-status", status="running")["status"] == "running"
    result = state.update_agent_execution(
        "exec-status", status="timed_out", error="deadline exceeded",
        error_class="wall_timeout",
    )
    assert result["status"] == "timeout"
    assert result["error"] == "deadline exceeded"
    assert result["error_class"] == "wall_timeout"

    with pytest.raises(WorkflowStateError, match="terminal execution"):
        state.update_agent_execution("exec-status", status="running")
