"""End-to-end tests for role-alias resolution through the registry to DIRECT_ANTHROPIC."""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from enhanced_router.app import app
from enhanced_router.backends import BackendType, ROLE_MODEL_ALIASES, sanitize_upstream_headers
from enhanced_router.routing import resolve_request
from enhanced_router.state import get_state
from enhanced_router.registry import get_registry


# Set API key env var so direct-anthropic credential check passes in tests
os.environ.setdefault("LONGCAT_API_KEY", "test-key-for-testing")


# ===================================================================
# Helpers — we manage a dedicated test DB so we don't pollute the real one
# ===================================================================

_TEST_DB = None


def _test_db_path():
    global _TEST_DB
    if _TEST_DB is None:
        import tempfile
        _TEST_DB = tempfile.mktemp(suffix=".db")
    return _TEST_DB


def _fresh_state():
    """Return a new RouteState backed by a temporary DB, replacing the singleton."""
    from enhanced_router.state import RouteState

    path = _test_db_path()
    if os.path.exists(path):
        os.unlink(path)
    state = RouteState(path)
    # Replace the module-level singleton so resolve_request sees it
    import enhanced_router.state as state_mod
    state_mod._state = state
    # Also refresh the registry singleton
    import enhanced_router.registry as reg_mod
    reg_mod._registry_instance = None
    get_registry()  # warm it
    return state


# ===================================================================
# test_role_alias_resolves_to_direct_anthropic
# ===================================================================


def test_role_alias_resolves_to_direct_anthropic():
    """brigade-implementer alias resolves to DIRECT_ANTHROPIC with longcat-2."""
    state = _fresh_state()
    run_id = "test-run-1"
    agent_id = "test-agent-1"
    state.create_run(run_id, "test-session", "/tmp")
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "longcat")

    result = resolve_request(
        "anthropic-brigade-implementer", run_id, agent_id
    )
    assert result.kind == BackendType.DIRECT_ANTHROPIC
    assert result.model_id == "longcat-2"
    assert result.upstream_model == "LongCat-2.0"
    assert result.agent_binding_id is not None
    assert result.route_version is not None
    assert result.registry_hash is not None


# ===================================================================
# test_role_alias_binding_is_immutable
# ===================================================================


def test_role_alias_binding_is_immutable():
    """Same agent gets same binding on second resolve."""
    state = _fresh_state()
    run_id = "test-run-2"
    agent_id = "test-agent-2"
    state.create_run(run_id, "test-session", "/tmp")
    state.create_epoch_from_profile(run_id, "ep-2", "normal", "longcat")

    r1 = resolve_request(
        "anthropic-brigade-implementer", run_id, agent_id
    )
    assert r1.kind == BackendType.DIRECT_ANTHROPIC
    assert r1.upstream_model == "LongCat-2.0"
    binding_id_1 = r1.agent_binding_id

    # Second resolve should return the same binding
    r2 = resolve_request(
        "anthropic-brigade-implementer", run_id, agent_id
    )
    assert r2.agent_binding_id == binding_id_1
    assert r2.upstream_model == "LongCat-2.0"
    assert r2.kind == BackendType.DIRECT_ANTHROPIC


# ===================================================================
# test_role_alias_requires_run_id
# ===================================================================


def test_role_alias_requires_run_id():
    """Role alias without run_id raises 409."""
    _fresh_state()
    with pytest.raises(Exception) as exc_info:
        resolve_request("anthropic-brigade-implementer", None, None)
    assert exc_info.value.status_code == 409


# ===================================================================
# test_role_alias_requires_agent_id
# ===================================================================


def test_role_alias_requires_agent_id():
    """Role alias without agent_id raises 409."""
    _fresh_state()
    with pytest.raises(Exception) as exc_info:
        resolve_request("anthropic-brigade-implementer", "run-1", None)
    assert exc_info.value.status_code == 409


# ===================================================================
# test_passthrough_model_remains_passthrough
# ===================================================================


def test_passthrough_model_remains_passthrough():
    """claude-sonnet-5 resolves to ANTHROPIC_PASSTHROUGH."""
    _fresh_state()
    result = resolve_request("claude-sonnet-5", None, None)
    assert result.kind == BackendType.ANTHROPIC_PASSTHROUGH


# ===================================================================
# test_sanitize_upstream_headers
# ===================================================================


def test_private_headers_not_leaked():
    """sanitize_upstream_headers strips internal headers."""
    clean = sanitize_upstream_headers({
        "x-enhanced-token": "secret",
        "x-brigade-run-id": "run-123",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "authorization": "Bearer sk-...",
    })
    assert "x-enhanced-token" not in clean
    assert "x-brigade-run-id" not in clean
    assert clean.get("anthropic-version") == "2023-06-01"
    assert clean.get("content-type") == "application/json"
    assert clean.get("authorization") == "Bearer sk-..."


# ===================================================================
# test_all_role_aliases_resolve
# ===================================================================


@pytest.mark.parametrize("alias,expected_role", [
    ("anthropic-brigade-recon", "recon"),
    ("anthropic-brigade-implementer", "implementer"),
    ("anthropic-brigade-adversary", "adversary"),
    ("anthropic-brigade-repairer", "repairer"),
])
def test_all_role_aliases_resolve(alias, expected_role):
    """Every known alias should resolve correctly."""
    state = _fresh_state()
    run_id = f"test-run-{expected_role}"
    agent_id = f"agent-{expected_role}"
    state.create_run(run_id, "test-session", "/tmp")
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "longcat")

    result = resolve_request(alias, run_id, agent_id)
    assert result.kind == BackendType.DIRECT_ANTHROPIC
    assert result.role == expected_role
    assert result.model_id == "longcat-2"
    assert result.agent_binding_id is not None


# ===================================================================
# test_resolve_with_existing_binding_returns_pinned_fields
# ===================================================================


def test_resolve_with_existing_binding_returns_pinned_fields():
    """Second resolve should carry backend/registry_hash from the binding."""
    state = _fresh_state()
    run_id = "test-run-pinned"
    agent_id = "pinned-agent-1"
    state.create_run(run_id, "test-session", "/tmp")
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "longcat")

    r1 = resolve_request("anthropic-brigade-implementer", run_id, agent_id)
    binding_id = r1.agent_binding_id

    # Second resolve
    r2 = resolve_request("anthropic-brigade-implementer", run_id, agent_id)
    assert r2.agent_binding_id == binding_id
    assert r2.registry_hash is not None
    assert r2.route_version is not None
