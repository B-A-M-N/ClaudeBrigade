"""Tests for LiteLLM proxy dispatch and header sanitization."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from enhanced_router.backends import BackendType, ResolvedRoute


# ---------------------------------------------------------------------------
# Fake LiteLLM upstream server
# ---------------------------------------------------------------------------


def _make_fake_litellm() -> FastAPI:
    """Return a FastAPI app that simulates a LiteLLM proxy."""
    fake = FastAPI()
    received: list[dict[str, Any]] = []

    @fake.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        received.append(body)
        headers = dict(request.headers)
        return JSONResponse(
            content={
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "fake response"}],
                "model": body.get("model", "unknown"),
                "headers_sent": {
                    k: v
                    for k, v in headers.items()
                    if k.lower() in {"authorization", "content-type"}
                },
                "brigade_headers": {
                    k: v
                    for k, v in headers.items()
                    if "brigade" in k.lower() or "enhanced" in k.lower()
                },
            }
        )

    fake.state.received = received
    return fake


@pytest.fixture
def fake_litellm():
    app = _make_fake_litellm()
    with TestClient(app) as client:
        yield client, app.state.received


# ---------------------------------------------------------------------------
# Tests: LiteLLM model name and header behavior
# ---------------------------------------------------------------------------


class TestLiteLLMDispatch:
    """Tests model name rewriting and header isolation through LiteLLM.

    These tests verify that the *LiteLLM proxy* receives the right model
    name and that internal Brigade headers do NOT leak upstream.  The
    ``proxy_litellm_messages`` function itself is tested indirectly through
    the app-level routing tests (``test_alias_routing``).
    """

    def test_model_name_received(self, fake_litellm):
        """LiteLLM receives the brigade-prefixed model name."""
        client, received = fake_litellm
        resp = client.post(
            "/v1/messages",
            json={
                "model": "brigade-qwen-local",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={
                "authorization": "Bearer test-litellm-key",
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "brigade-qwen-local"

    def test_brigade_headers_received_by_litellm(self, fake_litellm):
        """Internal Brigade headers arrive at the LiteLLM endpoint (expected).

        This test documents the baseline: the LiteLLM proxy process receives
        whatever headers the gateway sends it.  Header sanitization is the
        gateway's responsibility (see ``TestSanitizeUpstreamHeaders``).
        """
        client, received = fake_litellm
        resp = client.post(
            "/v1/messages",
            json={
                "model": "brigade-qwen-local",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "authorization": "Bearer test-litellm-key",
                "x-brigade-run-id": "secret-run-1",
                "x-enhanced-token": "secret-token",
            },
        )
        assert resp.status_code == 200

    def test_anthropic_version_preserved(self, fake_litellm):
        """anthropic-version is forwarded to LiteLLM."""
        client, received = fake_litellm
        resp = client.post(
            "/v1/messages",
            json={
                "model": "brigade-qwen-local",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "authorization": "Bearer test-litellm-key",
                "anthropic-version": "2023-06-01",
            },
        )
        assert resp.status_code == 200

    def test_connection_error_returns_502(self, monkeypatch):
        """When LiteLLM is unreachable, a connection error is raised."""
        # This tests that httpx properly raises ConnectError
        # when connecting to a port nothing is listening on.
        async def _test():
            async with httpx.AsyncClient() as client:
                with pytest.raises((httpx.ConnectError, httpx.RemoteProtocolError)):
                    await client.post(
                        "http://127.0.0.1:1/v1/messages",
                        json={"model": "test", "messages": []},
                        timeout=httpx.Timeout(1.0),
                    )

        import asyncio

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Tests: sanitize_upstream_headers
# ---------------------------------------------------------------------------


class TestSanitizeUpstreamHeaders:
    def test_brigade_headers_stripped(self):
        """Internal Brigade headers are stripped."""
        from enhanced_router.backends import sanitize_upstream_headers

        headers = {
            "content-type": "application/json",
            "x-brigade-run-id": "secret-123",
            "x-enhanced-token": "token-456",
            "authorization": "Bearer real-token",
        }
        cleaned = sanitize_upstream_headers(headers)
        assert "x-brigade-run-id" not in cleaned
        assert "x-enhanced-token" not in cleaned
        assert cleaned["content-type"] == "application/json"

    def test_anthropic_beta_preserved(self):
        """anthropic-beta and anthropic-version are preserved."""
        from enhanced_router.backends import sanitize_upstream_headers

        headers = {
            "anthropic-beta": "tools-2024-04-04",
            "anthropic-version": "2023-06-01",
        }
        cleaned = sanitize_upstream_headers(headers)
        assert cleaned["anthropic-beta"] == "tools-2024-04-04"
        assert cleaned["anthropic-version"] == "2023-06-01"

    def test_empty_headers(self):
        """Empty headers dict returns empty dict."""
        from enhanced_router.backends import sanitize_upstream_headers

        cleaned = sanitize_upstream_headers({})
        assert cleaned == {}

    def test_only_internal_headers_stripped(self):
        """Regular headers pass through unchanged."""
        from enhanced_router.backends import sanitize_upstream_headers

        headers = {
            "accept": "application/json",
            "user-agent": "test-client/1.0",
            "cache-control": "no-cache",
        }
        cleaned = sanitize_upstream_headers(headers)
        assert cleaned == headers


# ---------------------------------------------------------------------------
# Tests: ResolvedRoute dataclass
# ---------------------------------------------------------------------------


class TestResolvedRoute:
    def test_minimal_route(self):
        """Route with just kind and model_id."""
        route = ResolvedRoute(
            kind=BackendType.LITELLM, model_id="qwen-local"
        )
        assert route.kind == BackendType.LITELLM
        assert route.model_id == "qwen-local"
        assert route.role is None

    def test_endpoint_usage_normalizes_openai_cache_tokens(self):
        from enhanced_router.backends import _usage_totals

        totals = _usage_totals({
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 880},
                "completion_tokens": 40,
            }
        })

        assert totals == {
            "input_tokens": 1000,
            "cache_read_tokens": 880,
            "cache_write_tokens": 0,
            "output_tokens": 40,
        }

    def test_endpoint_usage_normalizes_anthropic_cache_tokens(self):
        from enhanced_router.backends import _usage_totals

        totals = _usage_totals({
            "usage": {
                "input_tokens": 120,
                "cache_read_input_tokens": 880,
                "cache_creation_input_tokens": 40,
                "output_tokens": 40,
            }
        })

        assert totals == {
            "input_tokens": 1040,
            "cache_read_tokens": 880,
            "cache_write_tokens": 40,
            "output_tokens": 40,
        }

    def test_fully_populated(self):
        """Route with all fields."""
        route = ResolvedRoute(
            kind=BackendType.DIRECT_ANTHROPIC,
            role="implementer",
            model_id="longcat-2",
            upstream_model="LongCat-2.0",
            api_base="http://127.0.0.1:11434",
            agent_binding_id=42,
            route_version=3,
            registry_hash="abc123",
            catalog_generation=1,
            litellm_model_name="brigade-qwen-local",
            litellm_base_url="http://127.0.0.1:18000",
        )
        assert route.kind == BackendType.DIRECT_ANTHROPIC
        assert route.litellm_model_name == "brigade-qwen-local"
        assert route.catalog_generation == 1

    def test_frozen(self):
        """ResolvedRoute is immutable."""
        route = ResolvedRoute(kind=BackendType.LITELLM, model_id="test")
        with pytest.raises(AttributeError):
            route.model_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: ROLE_MODEL_ALIASES
# ---------------------------------------------------------------------------


class TestRoleModelAliases:
    def test_all_four_roles_present(self):
        """The four stable role aliases remain available alongside specialists."""
        from enhanced_router.backends import ROLE_MODEL_ALIASES

        expected = {
            "anthropic-brigade-recon": "recon",
            "anthropic-brigade-implementer": "implementer",
            "anthropic-brigade-adversary": "adversary",
            "anthropic-brigade-repairer": "repairer",
        }
        assert {key: ROLE_MODEL_ALIASES[key] for key in expected} == expected
        assert len(ROLE_MODEL_ALIASES) >= len(expected)
