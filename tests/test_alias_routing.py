"""End-to-end tests for role-alias resolution through the registry to DIRECT_ANTHROPIC."""

from __future__ import annotations

import os
import pathlib
import pytest

from enhanced_router.backends import BackendType, RequestIdentity, sanitize_upstream_headers
from enhanced_router.routing import resolve_request


# Set API key env var so direct-anthropic credential check passes in tests
os.environ.setdefault("FREEINFERENCE_API_KEY", "test-key-for-testing")


# ===================================================================
# Helpers
# ===================================================================


def _identity(run_id: str | None = None, agent_id: str | None = None) -> RequestIdentity:
    """Build a RequestIdentity for testing."""
    return RequestIdentity(
        run_id=run_id,
        claude_session_id="test-session",
        claude_agent_id=agent_id,
        claude_parent_agent_id=None,
        endpoint_kind="/v1/messages",
    )

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
    registry = __import__("enhanced_router.registry", fromlist=["ModelRegistry"]).ModelRegistry(
        pathlib.Path(__file__).resolve().parent.parent / "config"
    )
    registry.reload()
    reg_mod._registry_instance = registry
    # Production FreeInference routes require capability-specific endpoint
    # certification. Seed a deterministic fixture certification and a local
    # LiteLLM deployment so these tests exercise the new logical-model path.
    registry_hash = registry.registry_hash()
    for model_id, spec in registry.models.items():
        if spec.provider_id != "freeinference":
            continue
        endpoint_ids = list(spec.endpoints) or ["default"]
        for endpoint_id in endpoint_ids:
            state.record_endpoint_certifications(
                provider_id="freeinference",
                model_id=model_id,
                endpoint_id=endpoint_id,
                configuration_hash=registry_hash,
                harness_version="test",
                protocol_version="test",
                evidence_digest="test",
                capabilities={"messages": True, "streaming": True, "tools": True},
            )
    generation = state.create_litellm_generation(
        registry_hash=registry_hash,
        model_count=len(registry.models),
        config_digest="test",
        reason="alias-routing-test",
    )
    deployment = state.register_litellm_deployment(generation, 18997, pid=99997)
    state.update_litellm_deployment(deployment, status="active")
    state.activate_litellm_generation(generation)
    return state


# ===================================================================
# test_role_alias_resolves_to_direct_anthropic
# ===================================================================


def test_role_alias_resolves_to_certified_freeinference_model():
    """The role alias resolves through the certified logical FreeInference model."""
    state = _fresh_state()
    run_id = "test-run-1"
    agent_id = "test-agent-1"
    state.create_run(run_id, "test-session", "/tmp")
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "freeinference")

    result = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model="anthropic-brigade-implementer",
    )
    assert result.kind == BackendType.LITELLM
    assert result.model_id == "kimi-k2.7-code"
    assert result.upstream_model == "openai/kimi-k2.7-code"
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
    state.create_epoch_from_profile(run_id, "ep-2", "normal", "freeinference")

    r1 = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model="anthropic-brigade-implementer",
    )
    assert r1.kind == BackendType.LITELLM
    assert r1.upstream_model == "openai/kimi-k2.7-code"
    binding_id_1 = r1.agent_binding_id

    # Second resolve should return the same binding
    r2 = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model="anthropic-brigade-implementer",
    )
    assert r2.agent_binding_id == binding_id_1
    assert r2.upstream_model == "openai/kimi-k2.7-code"
    assert r2.kind == BackendType.LITELLM


# ===================================================================
# test_role_alias_requires_run_id
# ===================================================================


def test_role_alias_requires_run_id():
    """Role alias without run_id raises 409."""
    _fresh_state()
    with pytest.raises(Exception) as exc_info:
        resolve_request(
            identity=_identity(run_id=None, agent_id="agent-1"),
            public_model="anthropic-brigade-implementer",
        )
    assert exc_info.value.status_code == 409


# ===================================================================
# test_role_alias_requires_agent_id
# ===================================================================


def test_role_alias_requires_agent_id():
    """Role alias without agent_id raises 409."""
    _fresh_state()
    with pytest.raises(Exception) as exc_info:
        resolve_request(
            identity=_identity(run_id="run-1", agent_id=None),
            public_model="anthropic-brigade-implementer",
        )
    assert exc_info.value.status_code == 409


# ===================================================================
# test_passthrough_model_remains_passthrough
# ===================================================================


def test_passthrough_model_remains_passthrough():
    """claude-sonnet-5 resolves to ANTHROPIC_PASSTHROUGH."""
    _fresh_state()
    result = resolve_request(
        identity=_identity(run_id=None, agent_id=None),
        public_model="claude-sonnet-5",
    )
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
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "freeinference")

    result = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model=alias,
    )
    assert result.kind == BackendType.LITELLM
    assert result.role == expected_role
    expected_models = {
        "recon": "qwen3.6-35b",
        "implementer": "kimi-k2.7-code",
        "adversary": "glm-5.1",
        "repairer": "glm-5-turbo",
    }
    assert result.model_id == expected_models[expected_role]
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
    state.create_epoch_from_profile(run_id, "ep-1", "normal", "freeinference")

    r1 = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model="anthropic-brigade-implementer",
    )
    binding_id = r1.agent_binding_id

    # Second resolve
    r2 = resolve_request(
        identity=_identity(run_id, agent_id),
        public_model="anthropic-brigade-implementer",
    )
    assert r2.agent_binding_id == binding_id
    assert r2.registry_hash is not None
    assert r2.route_version is not None
