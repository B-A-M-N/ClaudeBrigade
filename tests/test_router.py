import pytest
from fastapi.testclient import TestClient
from enhanced_router.app import (
    app,
    LONGCAT_UPSTREAM_ID,
    normalize_longcat_payload,
    _parse_retry_after,
)
import httpx


def test_longcat_normalization_rewrites_model_and_beta_fields():
    payload = {
        "model": "anthropic-longcat-2-0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": {"strict": True, "cache_control": "user-data"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "output_config": {"effort": "high"},
        "tools": [
            {
                "name": "Read",
                "description": "read",
                "input_schema": {
                    "type": "object",
                    "properties": {"strict": {"type": "boolean"}},
                },
                "strict": True,
                "defer_loading": True,
            }
        ],
    }
    result = normalize_longcat_payload(payload)
    assert result["model"] == LONGCAT_UPSTREAM_ID
    assert result["thinking"] == {"type": "enabled"}
    assert "context_management" not in result
    assert "output_config" not in result
    assert "cache_control" not in result["messages"][0]["content"][0]
    assert result["messages"][0]["content"][0]["content"]["cache_control"] == "user-data"
    assert result["messages"][0]["content"][0]["content"]["strict"] is True
    assert "strict" not in result["tools"][0]
    assert "defer_loading" not in result["tools"][0]
    assert "strict" in result["tools"][0]["input_schema"]["properties"]


def test_parse_retry_after_numeric_and_invalid():
    headers = httpx.Headers({"retry-after": "5"})
    assert _parse_retry_after(headers) == 5.0

    headers_invalid = httpx.Headers({"retry-after": "invalid-date"})
    assert _parse_retry_after(headers_invalid) is None

    headers_missing = httpx.Headers({})
    assert _parse_retry_after(headers_missing) is None


def test_parse_retry_after_http_date():
    from datetime import datetime, timedelta, timezone
    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    headers = httpx.Headers({"retry-after": http_date})
    res = _parse_retry_after(headers)
    assert res is not None
    assert 5.0 <= res <= 15.0


def test_router_healthz_and_models_endpoints(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("enhanced_router.app.ROUTER_TOKEN_ENV", "ENHANCED_ROUTER_TOKEN")
    monkeypatch.setenv("ENHANCED_ROUTER_TOKEN", "test-token")

    # Request without token should fail 401
    resp = client.get("/healthz")
    assert resp.status_code == 401

    # Request with valid token should succeed
    resp = client.get("/healthz", headers={"x-enhanced-token": "test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"  # healthz now returns richer data
    assert data["protocol_version"] == 2

    # daemon_metadata must be present for v2+
    meta = data["daemon_metadata"]
    assert "build_hash" in meta
    assert "registry_hash" in meta
    assert meta["protocol_version"] == 2
    assert "config_dir" in meta
    assert "pid" in meta
    assert isinstance(meta["pid"], int)

    # Models endpoint with token
    resp = client.get("/v1/models", headers={"x-enhanced-token": "test-token"})
    assert resp.status_code == 200
    models = resp.json()["data"]
    ids = {model["id"] for model in models}

    assert "claude-sonnet-5" in ids
    # Role aliases should be advertised instead of LongCat ID
    assert "anthropic-brigade-recon" in ids
    assert "anthropic-brigade-implementer" in ids
    assert "anthropic-brigade-adversary" in ids
    assert "anthropic-brigade-repairer" in ids
    assert not any(model_id.endswith("[1m]") for model_id in ids)


def test_router_head_probe(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr("enhanced_router.app.ROUTER_TOKEN_ENV", "ENHANCED_ROUTER_TOKEN")
    monkeypatch.setenv("ENHANCED_ROUTER_TOKEN", "test-token")
    resp = client.head("/", headers={"x-enhanced-token": "test-token"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_fastpath_route_can_detach_from_prompt_intake(monkeypatch):
    """Prompt intake receives a job receipt instead of waiting on inference."""
    from types import SimpleNamespace

    class FakeRegistry:
        fastpath = SimpleNamespace(
            enabled=True, modes=["route"], model_id="diffusiongemma",
            timeout_seconds=5,
        )

        @classmethod
        def resolve_fastpath(cls, sidecar_profile_id):
            return cls.fastpath

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id="freeinference")

    class FakeExecutor:
        async def invoke_detached(self, **kwargs):
            return {"execution_id": "fp-test"}

    monkeypatch.setattr("enhanced_router.registry.get_registry", lambda: FakeRegistry())
    monkeypatch.setattr("enhanced_router.sidecar_executor.get_sidecar_executor", lambda: FakeExecutor())
    import enhanced_router.app as app_module

    execution = await app_module._queue_fastpath_job(
        packet={
            "run_id": "run-1", "epoch_id": "ep-1",
            "intake_id": "intake-1", "proposal_id": "proposal-1",
        },
        mode="route",
        runner=app_module._run_fastpath_route,
    )

    assert execution["execution_id"] == "fp-test"


# ==================================================================
# Anthropic passthrough backend tests
# ==================================================================


def test_model_spec_accepts_anthropic_passthrough_backend():
    """ModelSpec with backend='anthropic-passthrough' and upstream_model validates."""
    from enhanced_router.config_models import ModelSpec, ModelCapabilities

    spec = ModelSpec(
        display_name="Claude Sonnet 5",
        backend="anthropic-passthrough",
        upstream_model="claude-sonnet-5",
        capabilities=ModelCapabilities(
            tools=True,
            mutation=False,
            context_tokens=200000,
            reasoning="high",
            local=False,
        ),
        allowed_roles={"adversary", "recon"},
    )
    assert spec.backend == "anthropic-passthrough"
    assert spec.upstream_model == "claude-sonnet-5"


def test_model_spec_rejects_anthropic_passthrough_without_upstream_model():
    """ModelSpec with backend='anthropic-passthrough' but no upstream_model raises."""
    from pydantic import ValidationError

    from enhanced_router.config_models import ModelSpec, ModelCapabilities

    with pytest.raises(ValidationError, match="upstream_model"):
        ModelSpec(
            display_name="Bad Model",
            backend="anthropic-passthrough",
            upstream_model=None,
            capabilities=ModelCapabilities(
                tools=True,
                mutation=False,
                context_tokens=100000,
                reasoning="medium",
                local=False,
            ),
        )


def test_backend_kind_maps_anthropic_passthrough():
    """_backend_kind('anthropic-passthrough') returns ANTHROPIC_PASSTHROUGH."""
    from enhanced_router.routing import _backend_kind
    from enhanced_router.backends import BackendType

    kind = _backend_kind("anthropic-passthrough")
    assert kind == BackendType.ANTHROPIC_PASSTHROUGH


def test_backend_kind_raises_for_unknown_backend():
    """_backend_kind raises UnsupportedBackendError for unknown backend values."""
    from enhanced_router.routing import _backend_kind
    from enhanced_router.backends import UnsupportedBackendError

    with pytest.raises(UnsupportedBackendError, match="bogus-backend"):
        _backend_kind("bogus-backend")


def test_resolved_route_from_binding_raises_for_unknown_backend():
    """resolved_route_from_binding raises UnsupportedBackendError for unknown backend."""
    from enhanced_router.routing import resolved_route_from_binding
    from enhanced_router.backends import UnsupportedBackendError

    binding = {
        "backend": "nonexistent-backend",
        "role": "recon",
        "model_id": "m1",
    }
    with pytest.raises(UnsupportedBackendError, match="nonexistent-backend"):
        resolved_route_from_binding(binding)


def test_anthropic_passthrough_worker_proxy_exists():
    """proxy_anthropic_passthrough_worker is importable from backends."""
    from enhanced_router.backends import proxy_anthropic_passthrough_worker
    assert callable(proxy_anthropic_passthrough_worker)


def test_sanitize_upstream_headers_preserves_authorization():
    """sanitize_upstream_headers keeps authorization but strips brigade internals."""
    from enhanced_router.backends import sanitize_upstream_headers

    headers = {
        "authorization": "Bearer claude-oauth-token",
        "x-api-key": "sk-ant-123",
        "x-enhanced-token": "router-token",
        "x-brigade-run-id": "run-abc",
        "anthropic-version": "2023-06-01",
    }
    cleaned = sanitize_upstream_headers(headers)
    assert "authorization" in cleaned
    assert cleaned["authorization"] == "Bearer claude-oauth-token"
    assert "x-api-key" in cleaned
    assert "x-enhanced-token" not in cleaned
    assert "x-brigade-run-id" not in cleaned
    assert "anthropic-version" in cleaned


# ==================================================================
# Provider auth spec tests (P0-5)
# ==================================================================


def test_provider_auth_spec_direct_anthropic():
    """resolve_provider_auth returns x-api-key header for direct-anthropic backends."""
    from enhanced_router.backends import resolve_provider_auth, ProviderAuthSpec
    from enhanced_router.config_models import ModelAuthSpec

    auth_spec = ModelAuthSpec(type="x-api-key", header="x-api-key", prefix="")
    auth = resolve_provider_auth("ANTHROPIC_API_KEY", auth_spec)
    assert isinstance(auth, ProviderAuthSpec)
    assert auth.type == "x-api-key"
    assert auth.env == "ANTHROPIC_API_KEY"
    assert auth.header_name == "x-api-key"


def test_provider_auth_spec_none():
    """resolve_provider_auth returns 'none' when no api_key_env is given."""
    from enhanced_router.backends import resolve_provider_auth, ProviderAuthSpec

    auth = resolve_provider_auth(None, None)
    assert isinstance(auth, ProviderAuthSpec)
    assert auth.type == "none"
    assert auth.env is None
    assert auth.header_name is None


def test_provider_auth_spec_litellm():
    """resolve_provider_auth returns bearer + authorization for litellm backends."""
    from enhanced_router.backends import resolve_provider_auth, ProviderAuthSpec
    from enhanced_router.config_models import ModelAuthSpec

    auth_spec = ModelAuthSpec(type="bearer", header="authorization", prefix="Bearer ")
    auth = resolve_provider_auth("MY_API_KEY", auth_spec)
    assert isinstance(auth, ProviderAuthSpec)
    assert auth.type == "bearer"
    assert auth.env == "MY_API_KEY"
    assert auth.header_name == "authorization"


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json", "Connection": "close"},
        {"TRANSFER-ENCODING": "chunked", "x-request-id": "req-1"},
    ],
)
def test_copy_response_headers_strips_hop_by_hop_headers(headers):
    import httpx
    from enhanced_router.backends import copy_response_headers

    copied = copy_response_headers(httpx.Headers(headers))
    assert "content-type" in copied or "x-request-id" in copied
    assert all(key.lower() not in {"connection", "transfer-encoding"} for key in copied)


def test_incremental_sse_usage_parser_handles_split_events():
    from enhanced_router.backends import _consume_sse_usage_lines

    buffer = bytearray()
    usage = None
    for chunk in (b'data: {"u', b'sage": {"input_tokens": 7}}', b'\r\n', b'data: [DONE]\r\n'):
        usage, overflowed = _consume_sse_usage_lines(buffer, chunk, usage)
        assert overflowed is False
    assert usage == {"usage": {"input_tokens": 7}, "usage_complete": False}


def test_incremental_sse_usage_parser_merges_nested_anthropic_usage():
    from enhanced_router.backends import _consume_sse_usage_lines

    buffer = bytearray()
    usage = None
    usage, overflowed = _consume_sse_usage_lines(
        buffer,
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":11,"cache_read_input_tokens":4}}}\n',
        usage,
    )
    assert not overflowed
    usage, overflowed = _consume_sse_usage_lines(
        buffer,
        b'data: {"type":"message_delta","delta":{"usage":{"output_tokens":3}}}\n',
        usage,
    )
    assert not overflowed
    assert usage == {
        "usage": {
            "input_tokens": 11,
            "cache_read_input_tokens": 4,
            "output_tokens": 3,
        }
    }
    assert buffer == bytearray()


def test_incremental_sse_usage_parser_discards_oversized_unterminated_line():
    from enhanced_router.backends import _consume_sse_usage_lines

    buffer = bytearray()
    usage, overflowed = _consume_sse_usage_lines(
        buffer,
        b"data: " + b"x" * 32,
        None,
        max_line_bytes=16,
    )
    assert usage is None
    assert overflowed is True
    assert buffer == bytearray()
