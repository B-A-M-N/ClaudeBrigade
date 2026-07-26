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
    monkeypatch.setattr("enhanced_router.app.ROUTER_TOKEN", "test-token")

    # Request without token should fail 401
    resp = client.get("/healthz")
    assert resp.status_code == 401

    # Request with valid token should succeed
    resp = client.get("/healthz", headers={"x-enhanced-token": "test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"  # healthz now returns richer data

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
    monkeypatch.setattr("enhanced_router.app.ROUTER_TOKEN", "test-token")
    resp = client.head("/", headers={"x-enhanced-token": "test-token"})
    assert resp.status_code == 200
