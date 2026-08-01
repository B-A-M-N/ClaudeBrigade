from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
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
    ROLE_MODEL_BINDINGS,
    parse_request_identity,
    proxy_direct_anthropic as _proxy_direct_anthropic_backend,
    proxy_litellm_messages as _proxy_litellm_messages_backend,
    proxy_anthropic_passthrough_worker as _proxy_anthropic_passthrough_worker_backend,
    close_upstream_client,
    copy_response_headers,
    get_upstream_client,
)
from enhanced_router.base import HOP_BY_HOP
from enhanced_router.mcp_control import control_mcp
from enhanced_router.mcp_transport import authenticated_mcp_app
from enhanced_router.routing import resolve_request
from enhanced_router.state import get_state

LOGGER = logging.getLogger("claude-enhanced-router")
logging.basicConfig(level=os.getenv("ENHANCED_ROUTER_LOG_LEVEL", "INFO"))

ANTHROPIC_UPSTREAM = os.getenv("ANTHROPIC_UPSTREAM", "https://api.anthropic.com").rstrip("/")
LONGCAT_UPSTREAM = os.getenv("LONGCAT_UPSTREAM", "https://api.longcat.chat/anthropic").rstrip("/")
LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
ROUTER_TOKEN_ENV = "ENHANCED_ROUTER_TOKEN"
LONGCAT_PUBLIC_ID = os.getenv("LONGCAT_PUBLIC_ID", "anthropic-longcat-2-0")
LONGCAT_UPSTREAM_ID = os.getenv("LONGCAT_UPSTREAM_ID", "LongCat-2.0")

async def _init_litellm_supervisor(app: FastAPI) -> None:
    """Initialise the LiteLLM supervisor if BRIGADE_LITELLM_KEY is set.

    The supervisor manages LiteLLM child process lifecycle (generations).
    Once initialized, it is registered with the MCP control layer for
    ``reload_catalog`` tool support.
    """
    litellm_key = os.environ.get("BRIGADE_LITELLM_KEY", "")
    if not litellm_key:
        LOGGER.info("BRIGADE_LITELLM_KEY not set; LiteLLM supervisor disabled")
        return

    from enhanced_router.litellm_supervisor import LiteLLMSupervisor
    from enhanced_router.mcp_control import set_litellm_supervisor
    from enhanced_router.backends import configure_litellm_supervisor
    from enhanced_router.base import BRIGADE_CACHE_DIR

    from enhanced_router.registry import get_registry
    from enhanced_router.litellm_config import generate_litellm_config
    state = get_state()
    supervisor = LiteLLMSupervisor(state, BRIGADE_CACHE_DIR, litellm_key)
    set_litellm_supervisor(supervisor)
    configure_litellm_supervisor(supervisor)
    app.state.litellm_supervisor = supervisor

    # Create initial generation from current registry
    try:
        registry = get_registry()
        reg_hash = registry.registry_hash()
        config_text = generate_litellm_config(registry.models)

        if registry.models and any(
            s.has_litellm_endpoint() and s.enabled
            for s in registry.models.values()
        ):
            await supervisor.start_generation(
                registry_hash=reg_hash,
                models=registry.models,
                config_text=config_text,
                reason="app-startup",
            )
        else:
            LOGGER.info("No enabled litellm models; supervisor ready but idle")
    except Exception as exc:
        LOGGER.warning("LiteLLM initial generation failed: %s", exc)


async def _shutdown_litellm(app: FastAPI) -> None:
    """Shut down the LiteLLM supervisor and all child processes."""
    supervisor = getattr(app.state, "litellm_supervisor", None)
    if supervisor is not None:
        LOGGER.info("Shutting down LiteLLM supervisor")
        try:
            await supervisor.shutdown()
        except Exception as exc:
            LOGGER.warning("LiteLLM shutdown error: %s", exc)
    from enhanced_router.backends import configure_litellm_supervisor
    configure_litellm_supervisor(None)


def _configure_provider_admission() -> None:
    """Load provider limits before any request can enter a backend."""
    from enhanced_router.backends import configure_provider_admission
    from enhanced_router.provider_admission import ProviderLimits, apply_concurrency_env_override
    from enhanced_router.registry import get_registry

    registry = get_registry()
    configured: dict[str, ProviderLimits] = {}
    for provider_id, provider in registry.providers.items():
        limits = ProviderLimits(**provider.limits.model_dump())
        limits = apply_concurrency_env_override(limits, provider.max_concurrency_env)
        if provider.max_concurrency_env and os.getenv(provider.max_concurrency_env):
            LOGGER.info(
                "provider=%s concurrency overridden to %s",
                provider_id,
                limits.max_concurrency,
            )
        configured[provider_id] = limits
    configure_provider_admission(configured)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_provider_admission()
    try:
        state = get_state()
        lifecycle = state.reconcile_lifecycle()
        LOGGER.info("lifecycle reconciliation: %s", lifecycle)
        from enhanced_router.shadow_worktree import ShadowWorktreeManager
        integration = ShadowWorktreeManager.reconcile_pending_integrations(state)
        LOGGER.info("integration reconciliation: %s", integration)
    except Exception as exc:
        # Startup remains available for inspection, but readiness/health can
        # report the reconciliation failure rather than silently losing it.
        LOGGER.error("state reconciliation failed: %s", exc)
        app.state.reconciliation_error = str(exc)
    # Initialize LiteLLM supervisor if key is configured
    await _init_litellm_supervisor(app)
    # The mounted Streamable HTTP MCP application owns its request/session
    # lifecycle.  Do not enter its session manager here: doing so blocks
    # ordinary FastAPI TestClient/startup and couples provider routing to the
    # optional control surface.
    try:
        yield
    finally:
        from enhanced_router.sidecar_executor import shutdown_sidecar_executor
        await shutdown_sidecar_executor()
        await _shutdown_litellm(app)
        await close_upstream_client()


app = FastAPI(title="Claude Enhanced Router", docs_url=None, redoc_url=None, lifespan=lifespan)


# ---------------------------------------------------------------------------
# MCP mount
# ---------------------------------------------------------------------------

_mcp_asgi = authenticated_mcp_app(control_mcp.streamable_http_app(), None)
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
    token = os.getenv(ROUTER_TOKEN_ENV, "")
    if token and request.headers.get("x-enhanced-token") != token:
        raise HTTPException(status_code=401, detail="invalid router token")


@app.post("/internal/fastpath/route")
async def internal_fastpath_route(request: Request) -> JSONResponse:
    """Run the optional advisory fastpath on bounded loopback input."""
    _require_local(request)
    from enhanced_router.fastpath import FastpathClient, FastpathLimits, FastpathPolicyValidator, FastpathRouteProposal
    from enhanced_router.registry import get_registry
    from enhanced_router.backends import _provider_admission

    registry = get_registry()
    config = registry.fastpath
    if config is None or not config.enabled or "route" not in config.modes:
        raise HTTPException(status_code=404, detail="fastpath route mode is disabled")
    try:
        packet = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="fastpath packet must be JSON") from exc
    if not isinstance(packet, dict):
        raise HTTPException(status_code=400, detail="fastpath packet must be an object")
    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > config.max_packet_bytes:
        raise HTTPException(status_code=413, detail="fastpath packet exceeds byte limit")

    model = registry.get_model(config.model_id)
    provider_id = model.provider_id
    if not provider_id:
        raise HTTPException(status_code=503, detail="fastpath model has no provider")
    try:
        from enhanced_router.endpoint_selection import select_endpoint
        selected = select_endpoint(
            config.model_id, model, get_state(), explicit_endpoint=config.endpoint,
            require_certified=True, provider_id=provider_id,
            configuration_hash=registry.registry_hash(),
            # FastpathClient uses a bounded non-streaming JSON completion;
            # requiring streaming here made a valid structured endpoint look
            # uncertified and silently disabled the advisory lane.
            required_capabilities=("messages",),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"fastpath endpoint is not certified: {exc}") from exc

    client = FastpathClient(
        api_base=(selected.spec.api_base or model.api_base or "").rstrip("/"),
        model=(selected.spec.litellm_model or model.litellm_model or config.model_id).removeprefix("openai/"),
        api_key_env=selected.spec.api_key_env or model.api_key_env or "FREEINFERENCE_API_KEY",
        limits=FastpathLimits(
            timeout_seconds=config.timeout_seconds,
            max_packet_bytes=config.max_packet_bytes,
        ),
        system_prompts=config.system_prompts,
        max_output_tokens=config.max_output_tokens,
    )
    request_id = f"fastpath:{packet.get('intake_id', 'unknown')}"
    try:
        await _provider_admission.acquire_request(
            provider_id, request_id, deadline=asyncio.get_running_loop().time() + config.timeout_seconds
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="fastpath provider capacity unavailable") from exc
    try:
        result = await client.request("route", packet)
        proposal = FastpathRouteProposal.model_validate(result)
        minimum = str(packet.get("deterministic_minimum_tier", "normal"))
        FastpathPolicyValidator().validate_route(
            proposal, minimum_tier=minimum, registry=registry, state=get_state(),
            configuration_hash=registry.registry_hash(),
            confidence_threshold=config.route_confidence_threshold,
        )
        return JSONResponse({
            **proposal.model_dump(),
            "validation_status": "accepted_for_controller_review",
            "fastpath_model_id": config.model_id,
            "fastpath_endpoint_id": selected.endpoint_id,
        })
    except Exception as exc:
        if config.failure_policy == "fail":
            raise HTTPException(status_code=502, detail=f"fastpath validation failed: {exc}") from exc
        return JSONResponse({
            "validation_status": "bypassed",
            "validation_reason": str(exc)[:500],
            "fastpath_model_id": config.model_id,
            "confidence": 0.0,
        })
    finally:
        await _provider_admission.release_request(request_id)


@app.post("/internal/fastpath/verify")
async def internal_fastpath_verify(request: Request) -> JSONResponse:
    """Run advisory fastpath verification over an authoritative evidence packet.

    The endpoint deliberately returns a recommendation only.  It never marks
    a workflow phase, finding, or completion state as satisfied.  Callers may
    provide ``run_id``/``epoch_id`` to persist the recommendation for audit.
    """
    _require_local(request)
    from enhanced_router.fastpath import (
        FastpathClient,
        FastpathLimits,
        FastpathPolicyValidator,
        FastpathVerification,
    )
    from enhanced_router.registry import get_registry
    from enhanced_router.backends import _provider_admission

    registry = get_registry()
    config = registry.fastpath
    if config is None or not config.enabled or "verify" not in config.modes:
        raise HTTPException(status_code=404, detail="fastpath verify mode is disabled")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="fastpath packet must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="fastpath packet must be an object")
    packet = body.get("packet", body)
    if not isinstance(packet, dict):
        raise HTTPException(status_code=400, detail="fastpath packet must be an object")
    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > config.max_packet_bytes:
        raise HTTPException(status_code=413, detail="fastpath packet exceeds byte limit")

    model = registry.get_model(config.model_id)
    provider_id = model.provider_id
    if not provider_id:
        raise HTTPException(status_code=503, detail="fastpath model has no provider")
    try:
        from enhanced_router.endpoint_selection import select_endpoint

        selected = select_endpoint(
            config.model_id,
            model,
            get_state(),
            explicit_endpoint=config.endpoint,
            require_certified=True,
            provider_id=provider_id,
            configuration_hash=registry.registry_hash(),
            required_capabilities=("messages",),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"fastpath endpoint is not certified: {exc}") from exc

    client = FastpathClient(
        api_base=(selected.spec.api_base or model.api_base or "").rstrip("/"),
        model=(selected.spec.litellm_model or model.litellm_model or config.model_id).removeprefix("openai/"),
        api_key_env=selected.spec.api_key_env or model.api_key_env or "FREEINFERENCE_API_KEY",
        limits=FastpathLimits(timeout_seconds=config.timeout_seconds, max_packet_bytes=config.max_packet_bytes),
        system_prompts=config.system_prompts,
        max_output_tokens=config.max_output_tokens,
    )
    request_id = f"fastpath-verify:{body.get('verification_id', body.get('epoch_id', 'unknown'))}"
    try:
        await _provider_admission.acquire_request(
            provider_id,
            request_id,
            deadline=asyncio.get_running_loop().time() + config.timeout_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="fastpath provider capacity unavailable") from exc
    try:
        result = await client.request("verify", packet)
        verification = FastpathVerification.model_validate(result)
        FastpathPolicyValidator().validate_verification(
            verification,
            packet=packet,
            deterministic_failed=bool(body.get("deterministic_failed", False)),
        )
        response = {
            **verification.model_dump(),
            "validation_status": "advisory",
            "fastpath_model_id": config.model_id,
            "fastpath_endpoint_id": selected.endpoint_id,
        }
        run_id = body.get("run_id")
        epoch_id = body.get("epoch_id")
        if isinstance(run_id, str) and isinstance(epoch_id, str):
            state = get_state()
            verification_id = str(body.get("verification_id") or f"verify:{request_id}")
            state.create_fastpath_verification(
                verification_id=verification_id,
                run_id=run_id,
                epoch_id=epoch_id,
                contract_digest=hashlib.sha256(
                    json.dumps(packet.get("contract", {}), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                evidence_digest=hashlib.sha256(encoded).hexdigest(),
                decision=verification.decision,
                checks_json=json.dumps(verification.checks, sort_keys=True, separators=(",", ":")),
                violations_json=json.dumps(verification.violations, separators=(",", ":")),
                confidence=verification.confidence,
                policy_disposition="advisory",
            )
            response["verification_id"] = verification_id
        return JSONResponse(response)
    except Exception as exc:
        if config.failure_policy == "fail":
            raise HTTPException(status_code=502, detail=f"fastpath verification failed: {exc}") from exc
        return JSONResponse({
            "decision": "escalate",
            "validation_status": "bypassed",
            "validation_reason": str(exc)[:500],
            "fastpath_model_id": config.model_id,
            "confidence": 0.0,
        })
    finally:
        await _provider_admission.release_request(request_id)


@app.post("/internal/catalog/sync")
async def internal_catalog_sync(request: Request) -> JSONResponse:
    """Synchronize an authenticated provider catalog on explicit request.

    Health and readiness checks never call the provider catalog. Existing
    bindings remain pinned; only new bindings see the published generation.
    """
    _require_local(request)
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="catalog sync requires a JSON object")
    provider_id = body.get("provider_id", "freeinference")
    endpoint_id = body.get("endpoint_id", "openai")
    if not isinstance(provider_id, str) or not isinstance(endpoint_id, str):
        raise HTTPException(status_code=400, detail="provider_id and endpoint_id must be strings")
    from enhanced_router.registry import get_registry

    registry = get_registry()
    try:
        digest, model_count = registry.discover_provider_catalog(
            provider_id,
            endpoint_id=endpoint_id,
            state=get_state(),
            request_id=request.headers.get("x-request-id"),
        )
    except (ValueError, OSError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=f"catalog discovery failed: {exc}") from exc
    return JSONResponse(
        status_code=200,
        content={
            "provider_id": provider_id,
            "endpoint_id": endpoint_id,
            "model_count": model_count,
            "response_digest": digest,
            "registry_hash": registry.registry_hash(),
        },
    )


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

    client: httpx.AsyncClient = get_upstream_client()

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
                await asyncio.sleep(max(0.1, _RETRY_BASE_DELAY * (2 ** attempt)))
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
            delay = max(0.1, min(delay, 30.0))
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

        response_headers = copy_response_headers(upstream_response.headers)
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

    return JSONResponse(
        status_code=502,
        content={"error": {"type": "upstream_request_error", "message": "no upstream response received after all retries"}},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.head("/")
async def root_probe(request: Request) -> Response:
    _require_local(request)
    return Response(status_code=200)


@app.get("/livez")
async def livez(request: Request) -> Response:
    _require_local(request)
    return Response(status_code=200)


@app.get("/readyz")
async def readyz(request: Request) -> Response:
    _require_local(request)
    try:
        from enhanced_router.registry import get_registry
        registry = get_registry()
        profile_id = os.environ.get("CLAUDE_BRIGADE_PROFILE", "hybrid")
        profile_readiness = registry.profile_readiness(profile_id)
        if not profile_readiness["ready"]:
            raise RuntimeError(
                f"profile '{profile_id}' is not ready: {profile_readiness['roles']}"
            )
        controller_models = registry.controller_models()
        if not controller_models:
            raise RuntimeError("no controller-eligible model is configured")
        if not any(spec.has_litellm_endpoint() for _, spec in controller_models) and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("no controller transport is ready")
        profile_models = [
            registry.get_model(registry.get_profile(profile_id).route_target(role).model)
            for role in ("recon", "implementer", "adversary", "repairer")
        ]
        if any(spec.has_litellm_endpoint() for spec in profile_models):
            supervisor = getattr(app.state, "litellm_supervisor", None)
            if supervisor is None or supervisor.active_generation is None:
                raise RuntimeError("profile requires an active LiteLLM generation")
        return JSONResponse({"status": "ready", "profile": profile_readiness})
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": str(exc)})


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    _require_local(request)
    try:
        from enhanced_router.registry import get_registry
        from enhanced_router.backends import provider_admission_snapshots
        registry = get_registry()
        registry_hash = registry.registry_hash()

        model_count = len(registry.models)
        litellm_count = sum(1 for s in registry.models.values() if s.has_litellm_endpoint() and s.enabled)
        healthy_count = sum(
            1 for model_id, spec in registry.models.items()
            if spec.enabled and (state_health := get_state().get_model_health(model_id))
            and state_health.get("status") == "healthy"
        )

        # LiteLLM status from supervisor
        litellm_status = "disabled"
        litellm_generation = None
        litellm_port = None
        supervisor = getattr(app.state, "litellm_supervisor", None)
        if supervisor is not None:
            litellm_status = getattr(supervisor, "lifecycle_state", "ready")
            litellm_generation = supervisor.active_generation
            litellm_port = supervisor.active_port
            if litellm_port is None:
                litellm_status = "degraded" if litellm_status == "active" else litellm_status

        default_profile = os.environ.get("CLAUDE_BRIGADE_PROFILE", "hybrid")
        profile_readiness = registry.profile_readiness(default_profile)

        from enhanced_router.base import BRIGADE_CONFIG_DIR

        health = {
            "status": "ok",
            "service": "claude-brigade",
            "protocol_version": 2,
            "gateway": "ok",
            "mcp": "ok",
            "registry": "ok",
            "litellm": litellm_status,
            "litellm_generation": litellm_generation,
            "litellm_port": litellm_port,
            "litellm_models_configured": litellm_count,
            "healthy_models": healthy_count,
            "configured_models": model_count,
            "profile": profile_readiness,
            "provider_admission": provider_admission_snapshots(),
            "registry_hash": registry_hash,
            "daemon_metadata": {
                "build_hash": "",
                "registry_hash": registry_hash,
                "protocol_version": 2,
                "config_dir": str(BRIGADE_CONFIG_DIR),
                "pid": os.getpid(),
            },
        }
    except Exception as exc:
        health = {
            "status": "degraded",
            "service": "claude-brigade",
            "protocol_version": 2,
            "gateway": "ok",
            "mcp": "ok",
            "registry": "error",
            "litellm": "disabled",
            "registry_error": str(exc),
            "daemon_metadata": {
                "build_hash": "",
                "registry_hash": "",
                "protocol_version": 2,
                "config_dir": "",
                "pid": os.getpid(),
            },
        }
    return health


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    _require_local(request)
    from enhanced_router.registry import get_registry

    registry = get_registry()
    entries: list[dict[str, Any]] = [
        {
            "id": model_id,
            "display_name": spec.display_name,
            "type": "model",
            "provider": spec.provider_id,
            "controller_eligible": spec.capabilities.controller_eligible,
            "status": "enabled" if spec.enabled else "disabled",
            "endpoints": {
                endpoint_id: {
                    "provider": endpoint.provider_id or spec.provider_id,
                    "backend": endpoint.backend,
                    "protocol": endpoint.protocol,
                    "certified": endpoint.certified,
                }
                for endpoint_id, endpoint in spec.endpoints.items()
            },
        }
        for model_id, spec in sorted(registry.models.items())
        if spec.enabled
    ]
    entries.extend(
        {
            "id": alias,
            "display_name": f"Brigade {role.title()}",
            "type": "model",
            "controller_eligible": False,
            "status": "healthy",
        }
        for alias, role in sorted(ROLE_MODEL_ALIASES.items())
    )
    entries.extend(
        {
            "id": alias,
            "display_name": f"Brigade {alias.removeprefix('anthropic-brigade-').replace('-', ' ').title()}",
            "type": "model",
            "provider": (
                registry.models[ROLE_MODEL_BINDINGS[alias]].provider_id
                if ROLE_MODEL_BINDINGS[alias] in registry.models
                else None
            ),
            "controller_eligible": False,
            "status": "healthy",
            "backing_model": ROLE_MODEL_BINDINGS[alias],
        }
        for alias in sorted(ROLE_MODEL_BINDINGS)
    )
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

    # Identity-first resolution: extract identity, then call resolve_request
    # which will look up existing bindings BEFORE examining the model string (P0-1).
    identity = parse_request_identity(request)

    # Controller model policy enforcement for non-alias models
    resolved: ResolvedRoute = resolve_request(
        identity=identity,
        public_model=model,
    )

    match resolved.kind:
            case BackendType.ANTHROPIC_PASSTHROUGH:
                if resolved.upstream_model:
                    payload["model"] = resolved.upstream_model
                return await _proxy_anthropic_passthrough_worker_backend(request, payload, resolved)

            case BackendType.DIRECT_ANTHROPIC:
                if not resolved.upstream_model:
                    raise HTTPException(
                        status_code=500,
                        detail=f"DIRECT_ANTHROPIC route for {model} has no upstream_model.",
                    )
                payload["model"] = resolved.upstream_model
                return await _proxy_direct_anthropic_backend(request, payload, resolved)

            case BackendType.LITELLM:
                if not resolved.upstream_model:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LITELLM route for {model} has no upstream_model.",
                    )
                payload["model"] = resolved.upstream_model
                return await _proxy_litellm_messages_backend(request, payload, resolved)

            case _:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unhandled backend kind: {resolved.kind}",
                )

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
