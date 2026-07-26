from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from enhanced_router.backends import (
    BackendType,
    ResolvedRoute,
    ROLE_MODEL_ALIASES,
    sanitize_upstream_headers,
    proxy_litellm_messages as _proxy_litellm_messages_backend,
)
from enhanced_router.mcp_control import control_mcp
from enhanced_router.mcp_transport import authenticated_mcp_app
from enhanced_router.routing import resolve_request
from enhanced_router.state import get_state

LOGGER = logging.getLogger("claude-enhanced-router")
logging.basicConfig(level=os.getenv("ENHANCED_ROUTER_LOG_LEVEL", "INFO"))

ANTHROPIC_UPSTREAM = os.getenv("ANTHROPIC_UPSTREAM", "https://api.anthropic.com").rstrip("/")
LONGCAT_UPSTREAM = os.getenv("LONGCAT_UPSTREAM", "https://api.longcat.chat/anthropic").rstrip("/")
LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
ROUTER_TOKEN = os.getenv("ENHANCED_ROUTER_TOKEN", "")
LONGCAT_PUBLIC_ID = os.getenv("LONGCAT_PUBLIC_ID", "anthropic-longcat-2-0")
LONGCAT_UPSTREAM_ID = os.getenv("LONGCAT_UPSTREAM_ID", "LongCat-2.0")

# Sonnet/Haiku IDs that are forwarded directly to Anthropic (not LongCat).
# Claude Code's model-discovery (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1)
# validates that the selected model appears in GET /v1/models, so we must
# include these here even though they are really served by Anthropic upstream.
_PASSTHROUGH_MODELS: list[dict[str, str]] = [
    {"id": "claude-sonnet-5",        "display_name": "Claude Sonnet 5"},
    {"id": "claude-sonnet-4-5",      "display_name": "Claude Sonnet 4.5"},
    {"id": "claude-haiku-4-5",       "display_name": "Claude Haiku 4.5"},
    {"id": "claude-opus-4-5",        "display_name": "Claude Opus 4.5"},
    {"id": "claude-opus-4-7",        "display_name": "Claude Opus 4.7"},
]

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
    "host",
}


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=3600.0, write=120.0, pool=30.0),
        follow_redirects=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = _build_client()
    # Run the MCP session manager inside the FastAPI lifespan
    async with control_mcp.session_manager.run():
        try:
            yield
        finally:
            await app.state.client.aclose()


app = FastAPI(title="Claude Enhanced Router", docs_url=None, redoc_url=None, lifespan=lifespan)


# ---------------------------------------------------------------------------
# MCP mount
# ---------------------------------------------------------------------------

_mcp_asgi = authenticated_mcp_app(control_mcp.streamable_http_app(), get_state())
app.mount("/mcp", _mcp_asgi)


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def _is_loopback(host: str | None) -> bool:
    return host in {"127.0.0.1", "::1", "localhost", "testclient", None}


def _require_local(request: Request) -> None:
    peer = request.client.host if request.client else None
    if not _is_loopback(peer):
        raise HTTPException(status_code=403, detail="loopback access only")
    if ROUTER_TOKEN and request.headers.get("x-enhanced-token") != ROUTER_TOKEN:
        raise HTTPException(status_code=401, detail="invalid router token")


# ---------------------------------------------------------------------------
# LongCat payload normalization (unchanged from original)
# ---------------------------------------------------------------------------


def _is_longcat(model: Any) -> bool:
    return isinstance(model, str) and model in {LONGCAT_PUBLIC_ID, LONGCAT_UPSTREAM_ID}


def _strip_cache_control_from_blocks(value: Any) -> Any:
    """Remove Anthropic prompt-cache markers without touching nested user/tool data."""
    if not isinstance(value, list):
        return value
    result: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            cleaned = dict(item)
            cleaned.pop("cache_control", None)
            result.append(cleaned)
        else:
            result.append(item)
    return result


def _normalize_messages(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    messages: list[Any] = []
    for message in value:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        cleaned = dict(message)
        cleaned["content"] = _strip_cache_control_from_blocks(cleaned.get("content"))
        messages.append(cleaned)
    return messages


def _normalize_tools(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    tools: list[Any] = []
    for tool in value:
        if not isinstance(tool, dict):
            tools.append(tool)
            continue
        cleaned = dict(tool)
        for unsupported in ("cache_control", "strict", "defer_loading"):
            cleaned.pop(unsupported, None)
        tools.append(cleaned)
    return tools


def normalize_longcat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate Claude Code's Anthropic dialect to LongCat's supported subset."""
    allowed = {
        "model",
        "messages",
        "system",
        "max_tokens",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "thinking",
        "stop_sequences",
        "metadata",
    }
    normalized = {key: value for key, value in payload.items() if key in allowed}
    normalized["model"] = LONGCAT_UPSTREAM_ID
    normalized["messages"] = _normalize_messages(normalized.get("messages"))

    system = normalized.get("system")
    if system is not None:
        normalized["system"] = _strip_cache_control_from_blocks(system)
    else:
        normalized.pop("system", None)

    tools = normalized.get("tools")
    if tools is not None:
        normalized["tools"] = _normalize_tools(tools)
    else:
        normalized.pop("tools", None)

    thinking = normalized.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type not in {"enabled", "disabled"}:
            normalized["thinking"] = {"type": "enabled"}
        else:
            normalized["thinking"] = {"type": thinking_type}

    return normalized


# ---------------------------------------------------------------------------
# Upstream request helpers
# ---------------------------------------------------------------------------


def _copy_request_headers(request: Request, *, longcat: bool) -> dict[str, str]:
    headers: dict[str, str] = {}
    BANNED = frozenset({"x-enhanced-token", "x-brigade-run-id"})
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP or lower in BANNED:
            continue
        if longcat:
            if lower in {"authorization", "x-api-key", "anthropic-beta"}:
                continue
            if lower.startswith("x-claude-code-"):
                continue
            if lower.startswith("anthropic-") and lower != "anthropic-version":
                continue
        headers[key] = value

    headers["content-type"] = "application/json"
    if longcat:
        if not LONGCAT_API_KEY or LONGCAT_API_KEY == "replace_me":
            raise HTTPException(status_code=503, detail="LongCat API key is not configured")
        headers["authorization"] = f"Bearer {LONGCAT_API_KEY}"
        headers.pop("x-api-key", None)
    return headers


def _copy_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP}


async def _stream_upstream(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        now = datetime.now(timezone.utc)
        diff = (dt - now).total_seconds()
        return diff if diff > 0 else None
    except Exception:
        return None


_RETRYABLE_STATUSES = {429, 503, 529}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5


async def _proxy(request: Request, payload: dict[str, Any], path: str) -> Response:
    longcat = _is_longcat(payload.get("model"))
    if longcat:
        payload = normalize_longcat_payload(payload)
        upstream = LONGCAT_UPSTREAM
    else:
        upstream = ANTHROPIC_UPSTREAM

    headers = _copy_request_headers(request, longcat=longcat)
    query = request.url.query
    url = f"{upstream}{path}" + (f"?{query}" if query else "")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    is_streaming = bool(payload.get("stream"))

    session_id = request.headers.get("x-claude-code-session-id", "-")
    agent_id = request.headers.get("x-claude-code-agent-id", "main")
    route = "longcat" if longcat else "anthropic"

    client: httpx.AsyncClient = request.app.state.client

    for attempt in range(_MAX_RETRIES):
        started = time.monotonic()
        try:
            upstream_request = client.build_request(request.method, url, headers=headers, content=body)
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "route=%s session=%s agent=%s attempt=%s upstream_error=%s",
                route, session_id, agent_id, attempt, type(exc).__name__,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return JSONResponse(
                status_code=502,
                content={"error": {"type": "upstream_connection_error", "message": str(exc)}},
            )

        elapsed_ms = int((time.monotonic() - started) * 1000)
        status = upstream_response.status_code

        if status in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES - 1:
            retry_after = _parse_retry_after(upstream_response.headers)
            delay = retry_after if retry_after is not None else (_RETRY_BASE_DELAY * (2 ** attempt))
            delay = min(delay, 30.0)
            await upstream_response.aclose()
            LOGGER.info(
                "route=%s session=%s agent=%s status=%s attempt=%s retrying_in=%.1fs",
                route, session_id, agent_id, status, attempt, delay,
            )
            await asyncio.sleep(delay)
            continue

        LOGGER.info(
            "route=%s model=%s session=%s agent=%s status=%s connect_ms=%s attempt=%s",
            route, payload.get("model"), session_id, agent_id, status, elapsed_ms, attempt,
        )

        response_headers = _copy_response_headers(upstream_response.headers)
        if is_streaming:
            if status in _RETRYABLE_STATUSES:
                LOGGER.warning(
                    "route=%s session=%s agent=%s retryable status=%s on streaming request; forwarding",
                    route, session_id, agent_id, status,
                )
            return StreamingResponse(
                _stream_upstream(upstream_response),
                status_code=status,
                headers=response_headers,
            )

        content = await upstream_response.aread()
        await upstream_response.aclose()
        return Response(content=content, status_code=status, headers=response_headers)

    return JSONResponse(status_code=502, content={"error": {"type": "upstream_connection_error", "message": "max retries exceeded"}})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.head("/")
async def root_probe(request: Request) -> Response:
    _require_local(request)
    return Response(status_code=200)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    _require_local(request)
    try:
        from enhanced_router.registry import ModelRegistry
        registry = ModelRegistry()
        registry.load_models()
        registry.load_profiles()
        registry.load_workflows()
        model_count = len(registry.models)
        health = {"status": "ok", "gateway": "ok", "mcp": "ok", "litellm": "not_configured"}
        health["registry_hash"] = registry.registry_hash()
        health["configured_models"] = model_count
        health["healthy_models"] = model_count
    except Exception as exc:
        health = {
            "status": "degraded",
            "gateway": "ok",
            "mcp": "ok",
            "litellm": "not_configured",
            "registry_error": str(exc),
        }
    return health


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    _require_local(request)
    entries: list[dict[str, Any]] = [
        {"id": alias, "display_name": f"Brigade {role.title()}", "type": "model"}
        for alias, role in ROLE_MODEL_ALIASES.items()
    ]
    for m in _PASSTHROUGH_MODELS:
        entries.append({"id": m["id"], "display_name": m["display_name"], "type": "model"})
    return {"data": entries, "has_more": False}


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    _require_local(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise HTTPException(status_code=400, detail="request requires a model")

    model = payload["model"]

    # Never forward a role alias upstream unchanged
    if model in ROLE_MODEL_ALIASES:
        resolved: ResolvedRoute = resolve_request(
            public_model=model,
            run_id=request.headers.get("x-brigade-run-id"),
            claude_agent_id=request.headers.get("x-claude-code-agent-id"),
        )

        # Build headers once, strip internal Brigade fields
        raw_headers = {k: v for k, v in request.headers.items()}
        clean_headers = sanitize_upstream_headers(raw_headers)

        match resolved.kind:
            case BackendType.ANTHROPIC_PASSTHROUGH:
                # A role alias resolving to passthrough is an internal error
                # unless the route explicitly targets an Anthropic API model.
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Role alias '{model}' resolved to unexpected passthrough "
                        f"for model '{resolved.model_id}'. Ensure the model is "
                        "properly configured in models.yaml."
                    ),
                )

            case BackendType.DIRECT_ANTHROPIC:
                if not resolved.upstream_model:
                    raise HTTPException(
                        status_code=500,
                        detail=f"DIRECT_ANTHROPIC route for {model} has no upstream_model.",
                    )
                payload["model"] = resolved.upstream_model
                if _is_longcat(resolved.upstream_model):
                    payload = normalize_longcat_payload(payload)
                return await _proxy(request, payload, "/v1/messages")

            case BackendType.LITELLM:
                if not resolved.upstream_model:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LITELLM route for {model} has no upstream_model.",
                    )
                payload["model"] = resolved.upstream_model
                # LiteLLM backend — uses sanitized headers, not _copy_request_headers
                return await _proxy_litellm(request, payload, resolved, clean_headers)

            case _:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unhandled backend kind: {resolved.kind}",
                )

    # Legacy LongCat direct model check
    if _is_longcat(model):
        return await _proxy(request, payload, "/v1/messages")

    # Standard Anthropic passthrough
    return await _proxy(request, payload, "/v1/messages")


async def _proxy_litellm(
    request: Request,
    payload: dict[str, Any],
    resolved: ResolvedRoute,
    headers: dict[str, str],
) -> Response:
    """Proxy a request through the internal LiteLLM child process.

    Delegates to backends.proxy_litellm_messages(). The role alias model ID
    has already been replaced with the LiteLLM model-group name in the
    resolve_request path.
    """
    return await _proxy_litellm_messages_backend(request, payload, resolved)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    _require_local(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise HTTPException(status_code=400, detail="request requires a model")
    model = payload["model"]

    # Role aliases — local estimation only
    if model in ROLE_MODEL_ALIASES:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "not_found_error",
                    "message": "Token counting for Brigade role aliases is unavailable; Claude Code should estimate locally",
                }
            },
        )

    # LongCat — also unavailable
    if _is_longcat(model):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "not_found_error",
                    "message": "LongCat count_tokens is unavailable; Claude Code should estimate locally",
                }
            },
        )

    return await _proxy(request, payload, "/v1/messages/count_tokens")