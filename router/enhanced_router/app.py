from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from enhanced_router.backends import (
    BackendType,
    ResolvedRoute,
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


def _max_request_bytes() -> int:
    """Return the bounded inbound JSON body limit.

    Claude requests can contain large tool transcripts, but an unbounded
    ``request.json()`` lets a disconnected or malicious client consume router
    memory before provider admission has a chance to protect the process.
    """
    raw = os.getenv("CLAUDE_BRIGADE_MAX_REQUEST_BYTES", "16777216").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 16 * 1024 * 1024
    return max(64 * 1024, min(value, 64 * 1024 * 1024))


async def _read_json_request(
    request: Request,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Read one bounded JSON request and fail before routing on overflow."""
    from json import JSONDecodeError

    limit = _max_request_bytes() if max_bytes is None else max(1024, int(max_bytes))
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
        if declared < 0:
            raise HTTPException(status_code=400, detail="invalid content-length")
        if declared > limit:
            raise HTTPException(status_code=413, detail="request body exceeds configured limit")
    body = await request.body()
    if len(body) > limit:
        raise HTTPException(status_code=413, detail="request body exceeds configured limit")
    try:
        payload = json.loads(body)
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return payload

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
        referenced_ids = registry.referenced_model_ids_for_active_runs(state)
        config_text = generate_litellm_config(registry.models, referenced_ids=referenced_ids)

        if registry.models and any(
            s.has_litellm_endpoint() and s.enabled
            for s in registry.models.values()
        ):
            await supervisor.start_generation(
                registry_hash=reg_hash,
                models=registry.models,
                config_text=config_text,
                reason="app-startup",
                referenced_ids=referenced_ids,
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


def _load_runtime_credentials() -> None:
    """Load provider credentials inside the router process only.

    The launcher deliberately starts the router without provider keys in its
    own environment.  The router may read the OS keyring, with the protected
    providers.env file retained as a compatibility fallback.
    """
    from enhanced_router.base import BRIGADE_CONFIG_DIR
    from enhanced_router.bootstrap_env import load_router_credentials

    result = load_router_credentials(BRIGADE_CONFIG_DIR)
    os.environ.update(result.provider_env)
    global LONGCAT_API_KEY
    LONGCAT_API_KEY = os.environ.get("LONGCAT_API_KEY", "")
    if result.provider_keys:
        LOGGER.info("loaded %d provider credential/config entries inside router", len(result.provider_keys))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_runtime_credentials()
    _configure_provider_admission()
    try:
        state = get_state()
        # A detached coprocessor has no child-process stop hook.  At router
        # startup, every still-active detached call belongs to the previous
        # process and must be surfaced immediately as an orphan, including a
        # job that was actively streaming when that process died.  Native
        # claims retain the normal age cutoff to avoid racing a live attach.
        lifecycle = state.reconcile_lifecycle(detached_max_age_seconds=0)
        expired_token_reservations = state.reconcile_token_reservations()
        feedback_reconciled = state.reconcile_feedback()
        lifecycle["token_reservations_reconciled"] = expired_token_reservations
        lifecycle["feedback_reconciled"] = feedback_reconciled
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
    """Queue or run the optional advisory fastpath as a sidecar execution."""
    _require_local(request)
    packet = await _read_json_request(request, max_bytes=64_000)
    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > 64_000:
        raise HTTPException(status_code=413, detail="fastpath packet exceeds byte limit")
    execution = await _queue_fastpath_job(
        packet=packet,
        mode="route",
        runner=lambda: _run_fastpath_route(packet),
    )
    if request.headers.get("x-brigade-fastpath-async") == "1":
        return JSONResponse({
            "validation_status": "queued",
            "execution_id": execution["execution_id"],
            "proposal_id": packet.get("proposal_id"),
            "intake_id": packet.get("intake_id"),
        }, status_code=202)
    return JSONResponse(await _wait_fastpath_job(str(execution["execution_id"])))


@app.post("/internal/feedback/checkpoint")
async def internal_feedback_checkpoint(request: Request) -> JSONResponse:
    """Evaluate one automatic hook checkpoint.

    This loopback endpoint is the hook-friendly projection of the same
    router-owned coprocessor control path exposed through MCP.  It performs no
    model call unless the persisted checkpoint policy admits one.
    """
    _require_local(request)
    body = await _read_json_request(request, max_bytes=64_000)
    encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > 64_000:
        raise HTTPException(status_code=413, detail="feedback checkpoint exceeds byte limit")
    from enhanced_router.feedback_service import request_feedback_checkpoint

    return JSONResponse(await request_feedback_checkpoint(state=get_state(), body=body))


def _resolve_run_fastpath(registry: Any, run_id: Any) -> Any:
    """Return the fastpath config for *run_id*'s selected sidecar profile.

    Fastpath is scoped per sidecar-profile rather than a bare global
    singleton (see ModelRegistry.resolve_fastpath): a run that selected a
    named sidecar profile gets that profile's own fastpath coprocessor;
    a run with no selection (or an unknown run_id) falls back to the
    global fastpath.yaml singleton, unchanged from prior behavior.
    """
    sidecar_profile_id = None
    if isinstance(run_id, str) and run_id:
        try:
            run_row = get_state().get_run(run_id)
        except Exception:
            # Prompt-intake callers may submit a receipt before the state
            # singleton is installed (and unknown run IDs use the global
            # fastpath by contract).  Do not turn that fallback lookup into a
            # database-path error.
            run_row = None
        if run_row:
            sidecar_profile_id = run_row.get("sidecar_profile_id")
    return registry.resolve_fastpath(sidecar_profile_id)


def _fastpath_route_candidates(config: Any) -> list[dict[str, Any]]:
    """Return the ordered, exact route candidates for a fastpath call."""
    from enhanced_router.route_ladder import candidate_dict, dedupe_candidates

    primary = {
        "model": config.model_id,
        "endpoint": getattr(config, "endpoint", "auto") or "auto",
        "provider_id": getattr(config, "provider_id", None),
    }
    fallback_routes = getattr(config, "fallback_routes", ()) or ()
    return dedupe_candidates(
        [candidate_dict(primary), *(candidate_dict(item) for item in fallback_routes)]
    )


def _fastpath_transport_failure(exc: BaseException) -> bool:
    """Classify failures that may safely advance a bounded route ladder.

    Structured-output failures are deliberately excluded: a malformed or
    policy-invalid answer is not evidence that another provider should be
    charged for the same advisory call.
    """
    if isinstance(exc, (httpx.HTTPError, TimeoutError, asyncio.TimeoutError, OSError)):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return not any(
            marker in text
            for marker in (
                "malformed json",
                "valid json",
                "must be an object",
                "did not contain json message content",
                "response was malformed",
            )
        )
    return False


def _fastpath_failure_summary(
    mode: str, failures: list[tuple[dict[str, Any], BaseException]]
) -> str:
    details = " | ".join(
        f"{item.get('provider_id') or 'default'}/{item.get('model')}@"
        f"{item.get('endpoint', 'auto')}: {exc}"
        for item, exc in failures
    )
    return f"fastpath {mode} route ladder exhausted" + (f": {details}" if details else "")


def _fastpath_attempt_identity(packet: dict[str, Any], mode: str, index: int) -> tuple[str, str]:
    """Return durable execution/attempt IDs for one fastpath candidate."""
    execution_id = str(packet.get("execution_id") or "").strip()
    if not execution_id:
        execution_id = "fastpath:" + hashlib.sha256(
            json.dumps(
                {"run_id": packet.get("run_id"), "epoch_id": packet.get("epoch_id"),
                 "mode": mode, "intake_id": packet.get("intake_id"),
                 "proposal_id": packet.get("proposal_id")},
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
    return execution_id, f"{execution_id}:route:{index}"


def _start_fastpath_attempt(
    *, state: Any, packet: dict[str, Any], mode: str, index: int,
    candidate: dict[str, Any],
) -> tuple[str, str]:
    execution_id, attempt_id = _fastpath_attempt_identity(packet, mode, index)
    try:
        from enhanced_router.route_ladder import candidate_dict, route_digest

        normalized = candidate_dict(candidate)
        state.start_route_attempt(
            attempt_id=attempt_id,
            run_id=str(packet["run_id"]),
            epoch_id=str(packet["epoch_id"]),
            execution_id=execution_id,
            candidate_index=index,
            model_id=normalized["model"],
            provider_id=normalized.get("provider_id"),
            endpoint_id=normalized.get("endpoint", "auto"),
            route_digest=route_digest(normalized),
        )
    except Exception:
        LOGGER.exception("failed to start fastpath route telemetry attempt=%s", attempt_id)
    return execution_id, attempt_id


def _finish_fastpath_attempt(
    *, state: Any, attempt_id: str, status: str, status_code: int | None = None,
    error_class: str | None = None, error: str | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    try:
        state.finish_route_attempt(
            attempt_id,
            status=status,
            status_code=status_code,
            error_class=error_class,
            error=error,
            usage=usage,
        )
    except Exception:
        LOGGER.exception("failed to finish fastpath route telemetry attempt=%s", attempt_id)


async def _queue_fastpath_job(
    *,
    packet: dict[str, Any],
    mode: str,
    runner: Any,
) -> dict[str, Any]:
    """Create one persisted advisory sidecar job and start its task."""
    from enhanced_router.registry import get_registry
    from enhanced_router.sidecar_executor import get_sidecar_executor

    registry = get_registry()
    config = _resolve_run_fastpath(registry, packet.get("run_id"))
    if config is None or not config.enabled or mode not in config.modes:
        raise HTTPException(status_code=404, detail=f"fastpath {mode} mode is disabled")
    run_id = str(packet.get("run_id") or "")
    epoch_id = str(packet.get("epoch_id") or "")
    if not run_id or not epoch_id:
        raise HTTPException(status_code=409, detail="fastpath requires an active run and epoch")
    queue_candidate: dict[str, Any] | None = None
    queue_provider_id: str | None = None
    for candidate in _fastpath_route_candidates(config):
        try:
            candidate_model = registry.get_model(str(candidate["model"]))
        except KeyError:
            continue
        candidate_provider = candidate.get("provider_id") or candidate_model.provider_id
        if getattr(candidate_model, "enabled", True) and candidate_provider:
            queue_candidate = candidate
            queue_provider_id = str(candidate_provider)
            break
    if queue_candidate is None or queue_provider_id is None:
        raise HTTPException(status_code=503, detail="fastpath route ladder has no usable provider")
    execution_id = f"fp_{uuid.uuid4().hex}"
    # The detached execution event is the durable recovery input.  Keep the
    # ID in the same packet object captured by the runner so route attempts
    # can be reconciled if the router dies between candidates.
    packet["execution_id"] = execution_id
    return await get_sidecar_executor().invoke_detached(
        run_id=run_id,
        epoch_id=epoch_id,
        execution_id=execution_id,
        phase_id=f"fastpath:{mode}",
        role="fastpath",
        model_id=str(queue_candidate["model"]),
        provider_id=queue_provider_id,
        packet=packet,
        timeout_seconds=config.timeout_seconds,
        runner=runner,
    )


async def _wait_fastpath_job(execution_id: str) -> dict[str, Any]:
    from enhanced_router.sidecar_executor import get_sidecar_executor

    execution = await get_sidecar_executor().wait(execution_id)
    if execution is None:
        raise HTTPException(status_code=502, detail="fastpath execution disappeared")
    if execution.get("status") != "completed":
        raise HTTPException(
            status_code=502,
            detail=str(execution.get("error") or "fastpath execution failed"),
        )
    try:
        result = json.loads(str(execution.get("result_json") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="fastpath result was malformed") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="fastpath result was not an object")
    return result


async def _run_fastpath_route(packet: dict[str, Any]) -> dict[str, Any]:
    """Execute one advisory route call after its request has been detached."""
    from enhanced_router.fastpath import (
        FastpathClient, FastpathLimits, FastpathPolicyValidator, FastpathRouteProposal,
        build_route_candidates, route_template,
    )
    from enhanced_router.registry import get_registry

    registry = get_registry()
    config = _resolve_run_fastpath(registry, packet.get("run_id"))
    if config is None or not config.enabled or "route" not in config.modes:
        raise HTTPException(status_code=404, detail="fastpath route mode is disabled")

    minimum_tier = str(packet.get("deterministic_minimum_tier", "normal"))
    roles = ["recon", "implementer", "adversary", "repairer"]
    packet_candidates, candidate_map = build_route_candidates(registry, roles)
    candidate_set_digest = hashlib.sha256(
        json.dumps(
            {
                "registry_hash": registry.registry_hash(),
                "candidates": packet_candidates,
                "minimum_tier": minimum_tier,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    packet = {
        **packet,
        "candidates": packet_candidates,
        "candidate_set_digest": candidate_set_digest,
        "registry_hash": registry.registry_hash(),
        "output_template": route_template(minimum_tier, roles),
    }

    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > config.max_packet_bytes:
        raise HTTPException(status_code=413, detail="fastpath packet exceeds byte limit")

    from enhanced_router.endpoint_selection import select_endpoint

    selected = None
    selected_candidate: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failures: list[tuple[dict[str, Any], BaseException]] = []
    for candidate in _fastpath_route_candidates(config):
        _execution_id, attempt_id = _start_fastpath_attempt(
            state=get_state(), packet=packet, mode="route",
            index=len(failures), candidate=candidate,
        )
        try:
            model = registry.get_model(str(candidate["model"]))
            provider_id = candidate.get("provider_id") or model.provider_id
            if not provider_id:
                raise RuntimeError("candidate has no provider")
            endpoint_id = candidate.get("endpoint", "auto")
            if endpoint_id == "auto":
                endpoint_id = None
            selected = select_endpoint(
                str(candidate["model"]), model, get_state(), explicit_endpoint=endpoint_id,
                require_certified=True, provider_id=str(provider_id),
                configuration_hash=registry.registry_hash(),
                # FastpathClient uses a bounded non-streaming JSON completion;
                # requiring streaming here made a valid structured endpoint look
                # uncertified and silently disabled the advisory lane.
                required_capabilities=("messages",),
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            _finish_fastpath_attempt(
                state=get_state(), attempt_id=attempt_id, status="failed",
                error_class="route_unavailable", error=str(exc),
            )
            failures.append((candidate, exc))
            continue

        client = FastpathClient(
            api_base=(selected.spec.api_base or model.api_base or "").rstrip("/"),
            model=(selected.spec.litellm_model or model.litellm_model or str(candidate["model"])).removeprefix("openai/"),
            api_key_env=selected.spec.api_key_env or model.api_key_env or "FREEINFERENCE_API_KEY",
            limits=FastpathLimits(
                timeout_seconds=config.timeout_seconds,
                max_packet_bytes=config.max_packet_bytes,
            ),
            provider_id=str(provider_id),
            endpoint_id=selected.endpoint_id,
            system_prompts=config.system_prompts,
            max_output_tokens=config.max_output_tokens,
            strict_schema=config.strict_schema,
            disable_thinking=config.disable_thinking,
        )
        try:
            result = await client.request("route", packet)
        except Exception as exc:
            if _fastpath_transport_failure(exc):
                _finish_fastpath_attempt(
                    state=get_state(), attempt_id=attempt_id, status="failed",
                    error_class="transport_failure", error=str(exc),
                )
                failures.append((candidate, exc))
                continue
            _finish_fastpath_attempt(
                state=get_state(), attempt_id=attempt_id, status="failed",
                error_class="provider_error", error=str(exc),
            )
            raise
        selected_candidate = candidate
        _finish_fastpath_attempt(
            state=get_state(), attempt_id=attempt_id, status="succeeded",
            usage=client.last_usage,
        )
        break

    if result is None or selected is None or selected_candidate is None:
        exc = RuntimeError(_fastpath_failure_summary("route", failures))
        if config.failure_policy == "fail":
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        result_payload = {
            "validation_status": "bypassed",
            "validation_reason": str(exc)[:500],
            "fastpath_model_id": config.model_id,
            "confidence": 0.0,
        }
    else:
        try:
            proposal = FastpathRouteProposal.model_validate(result)
            FastpathPolicyValidator().validate_route(
                proposal, minimum_tier=minimum_tier, registry=registry, state=get_state(),
                configuration_hash=registry.registry_hash(),
                confidence_threshold=config.route_confidence_threshold,
                candidate_map=candidate_map,
            )
            # Keep the compact candidate id for auditability, but also persist
            # the router-resolved model.  The controller must never have to
            # reconstruct DiffusionGemma's alias map after the proposal has
            # been detached from the packet.
            resolved_routes: dict[str, dict[str, Any]] = {}
            for role, target in proposal.routes.items():
                model_id = target.logical_model
                if target.candidate_id is not None:
                    model_id = candidate_map.get(role, {}).get(target.candidate_id)
                if model_id:
                    resolved_routes[role] = {
                        "model": model_id,
                        "endpoint": "auto",
                        "candidate_id": target.candidate_id,
                        "fanout": target.fanout,
                    }
            result_payload = {
                **proposal.model_dump(),
                "resolved_routes": resolved_routes,
                "validation_status": "accepted_for_controller_review",
                "fastpath_model_id": selected_candidate["model"],
                "fastpath_endpoint_id": selected.endpoint_id,
                "fastpath_provider_id": selected_candidate.get("provider_id"),
                "candidate_set_digest": candidate_set_digest,
                "minimum_tier": minimum_tier,
                "registry_hash": registry.registry_hash(),
            }
        except Exception as exc:
            _finish_fastpath_attempt(
                state=get_state(), attempt_id=attempt_id, status="failed",
                error_class="malformed_result", error=str(exc),
            )
            if config.failure_policy == "fail":
                raise HTTPException(status_code=502, detail=f"fastpath validation failed: {exc}") from exc
            result_payload = {
                "validation_status": "bypassed",
                "validation_reason": str(exc)[:500],
                "fastpath_model_id": selected_candidate["model"],
                "confidence": 0.0,
            }
    proposal_id = packet.get("proposal_id")
    intake_id = packet.get("intake_id")
    run_id = packet.get("run_id")
    if all(isinstance(value, str) and value for value in (proposal_id, intake_id, run_id)):
        try:
            get_state().create_route_proposal(
                proposal_id=str(proposal_id),
                intake_id=str(intake_id),
                source="fastpath",
                parsed_proposal=result_payload,
                validation_status=str(result_payload.get("validation_status", "bypassed")),
                validation_reason=str(result_payload.get("validation_reason", "")),
                fastpath_model_id=str(result_payload.get("fastpath_model_id", config.model_id)),
                fastpath_endpoint_id=str(result_payload.get("fastpath_endpoint_id", "")) or None,
                confidence=float(result_payload.get("confidence", 0.0) or 0.0),
            )
        except Exception:
            LOGGER.exception("failed to persist detached fastpath proposal=%s", proposal_id)
    return result_payload


@app.post("/internal/fastpath/verify")
async def internal_fastpath_verify(request: Request) -> JSONResponse:
    """Run advisory fastpath verification over an authoritative evidence packet.

    The endpoint deliberately returns a recommendation only.  It never marks
    a workflow phase, finding, or completion state as satisfied.  The request
    is represented as a persisted detached sidecar execution.
    """
    _require_local(request)
    from enhanced_router.registry import get_registry

    registry = get_registry()
    body = await _read_json_request(request, max_bytes=256_000)
    config = _resolve_run_fastpath(registry, body.get("run_id"))
    if config is None or not config.enabled or "verify" not in config.modes:
        raise HTTPException(status_code=404, detail="fastpath verify mode is disabled")
    packet = body.get("packet", body)
    if not isinstance(packet, dict):
        raise HTTPException(status_code=400, detail="fastpath packet must be an object")
    encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > config.max_packet_bytes:
        raise HTTPException(status_code=413, detail="fastpath packet exceeds byte limit")
    body_packet = dict(body)
    body_packet["packet"] = packet
    execution = await _queue_fastpath_job(
        packet=body_packet,
        mode="verify",
        runner=lambda: _run_fastpath_verify(body_packet),
    )
    if request.headers.get("x-brigade-fastpath-async") == "1":
        return JSONResponse(
            {
                "validation_status": "queued",
                "execution_id": execution["execution_id"],
                "verification_id": packet.get("verification_id"),
            },
            status_code=202,
        )
    return JSONResponse(await _wait_fastpath_job(str(execution["execution_id"])))


async def _run_fastpath_verify(body: dict[str, Any]) -> dict[str, Any]:
    """Execute and persist one advisory verification inside the sidecar lane."""
    from enhanced_router.fastpath import (
        FastpathClient,
        FastpathLimits,
        FastpathPolicyValidator,
        FastpathVerification,
    )
    from enhanced_router.registry import get_registry

    registry = get_registry()
    config = _resolve_run_fastpath(registry, body.get("run_id"))
    if config is None:
        raise RuntimeError("fastpath is not configured")
    packet = body["packet"]
    from enhanced_router.endpoint_selection import select_endpoint

    selected = None
    selected_candidate: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failures: list[tuple[dict[str, Any], BaseException]] = []
    for candidate in _fastpath_route_candidates(config):
        _execution_id, attempt_id = _start_fastpath_attempt(
            state=get_state(), packet=body, mode="verify",
            index=len(failures), candidate=candidate,
        )
        try:
            model = registry.get_model(str(candidate["model"]))
            provider_id = candidate.get("provider_id") or model.provider_id
            if not provider_id:
                raise RuntimeError("candidate has no provider")
            endpoint_id = candidate.get("endpoint", "auto")
            if endpoint_id == "auto":
                endpoint_id = None
            selected = select_endpoint(
                str(candidate["model"]), model, get_state(), explicit_endpoint=endpoint_id,
                require_certified=True, provider_id=str(provider_id),
                configuration_hash=registry.registry_hash(),
                required_capabilities=("messages",),
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            _finish_fastpath_attempt(
                state=get_state(), attempt_id=attempt_id, status="failed",
                error_class="route_unavailable", error=str(exc),
            )
            failures.append((candidate, exc))
            continue
        client = FastpathClient(
            api_base=(selected.spec.api_base or model.api_base or "").rstrip("/"),
            model=(selected.spec.litellm_model or model.litellm_model or str(candidate["model"])).removeprefix("openai/"),
            api_key_env=selected.spec.api_key_env or model.api_key_env or "FREEINFERENCE_API_KEY",
            limits=FastpathLimits(timeout_seconds=config.timeout_seconds, max_packet_bytes=config.max_packet_bytes),
            provider_id=str(provider_id),
            endpoint_id=selected.endpoint_id,
            system_prompts=config.system_prompts,
            max_output_tokens=config.max_output_tokens,
            strict_schema=config.strict_schema,
            disable_thinking=config.disable_thinking,
        )
        try:
            result = await client.request("verify", packet)
        except Exception as exc:
            if _fastpath_transport_failure(exc):
                _finish_fastpath_attempt(
                    state=get_state(), attempt_id=attempt_id, status="failed",
                    error_class="transport_failure", error=str(exc),
                )
                failures.append((candidate, exc))
                continue
            _finish_fastpath_attempt(
                state=get_state(), attempt_id=attempt_id, status="failed",
                error_class="provider_error", error=str(exc),
            )
            raise
        selected_candidate = candidate
        _finish_fastpath_attempt(
            state=get_state(), attempt_id=attempt_id, status="succeeded",
            usage=client.last_usage,
        )
        break

    if result is None or selected is None or selected_candidate is None:
        exc = RuntimeError(_fastpath_failure_summary("verify", failures))
        if config.failure_policy == "fail":
            raise RuntimeError(str(exc)) from exc
        return {
            "decision": "escalate",
            "validation_status": "bypassed",
            "validation_reason": str(exc)[:500],
            "fastpath_model_id": config.model_id,
            "confidence": 0.0,
        }

    try:
        verification = FastpathVerification.model_validate(result)
        FastpathPolicyValidator().validate_verification(
            verification,
            packet=packet,
        )
        response = {
            **verification.model_dump(),
            "validation_status": "advisory",
            "fastpath_model_id": selected_candidate["model"],
            "fastpath_endpoint_id": selected.endpoint_id,
            "fastpath_provider_id": selected_candidate.get("provider_id"),
        }
        run_id = body.get("run_id")
        epoch_id = body.get("epoch_id")
        if isinstance(run_id, str) and isinstance(epoch_id, str):
            state = get_state()
            verification_id = str(
                body.get("verification_id") or f"verify:{body.get('epoch_id', 'unknown')}"
            )
            state.create_fastpath_verification(
                verification_id=verification_id,
                run_id=run_id,
                epoch_id=epoch_id,
                contract_digest=hashlib.sha256(
                    json.dumps(packet.get("contract", {}), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                evidence_digest=hashlib.sha256(
                    json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode()
                ).hexdigest(),
                decision=verification.decision,
                checks_json=json.dumps(verification.checks, sort_keys=True, separators=(",", ":")),
                violations_json=json.dumps(verification.violations, separators=(",", ":")),
                confidence=verification.confidence,
                policy_disposition="advisory",
            )
            response["verification_id"] = verification_id
            if verification.decision in {"fail", "escalate"}:
                # Merge advice is deliberately not an integration authority,
                # but a late negative result must still become durable
                # controller-visible work instead of disappearing with the
                # background task.
                finding_id = f"fastpath:{verification_id}"
                if state.get_finding(finding_id) is None:
                    state.create_finding(
                        finding_id=finding_id,
                        run_id=run_id,
                        epoch_id=epoch_id,
                        description=(
                            "Fastpath merge verification requested controller review: "
                            + "; ".join(verification.violations[:8])
                        )[:4_000],
                        severity="high" if verification.decision == "fail" else "medium",
                        category="fastpath_merge_review",
                        source_phase_id="integration",
                        source_agent_id="coprocessor:fastpath",
                        evidence_json=json.dumps(
                            {
                                "verification_id": verification_id,
                                "decision": verification.decision,
                                "checks": verification.checks,
                                "violations": verification.violations,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
        return response
    except Exception as exc:
        _finish_fastpath_attempt(
            state=get_state(), attempt_id=attempt_id, status="failed",
            error_class="malformed_result", error=str(exc),
        )
        if config.failure_policy == "fail":
            raise RuntimeError(f"fastpath verification failed: {exc}") from exc
        return {
            "decision": "escalate",
            "validation_status": "bypassed",
            "validation_reason": str(exc)[:500],
            "fastpath_model_id": config.model_id,
            "confidence": 0.0,
        }


@app.post("/internal/catalog/sync")
async def internal_catalog_sync(request: Request) -> JSONResponse:
    """Synchronize an authenticated provider catalog on explicit request.

    Health and readiness checks never call the provider catalog. Existing
    bindings remain pinned; only new bindings see the published generation.
    """
    _require_local(request)
    body = await _read_json_request(request, max_bytes=64_000)
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
        litellm_telemetry: dict[str, Any] = {}
        supervisor = getattr(app.state, "litellm_supervisor", None)
        if supervisor is not None:
            litellm_status = getattr(supervisor, "lifecycle_state", "ready")
            litellm_generation = supervisor.active_generation
            litellm_port = supervisor.active_port
            if litellm_port is None:
                litellm_status = "degraded" if litellm_status == "active" else litellm_status
            snapshot = getattr(supervisor, "deployment_telemetry", None)
            if callable(snapshot):
                litellm_telemetry["runtime"] = snapshot()
            try:
                litellm_telemetry["deployments"] = get_state().get_litellm_deployment_telemetry(limit=20)
                litellm_telemetry["request_attributions"] = get_state().get_litellm_request_attributions(limit=50)
            except Exception:
                LOGGER.debug("LiteLLM deployment telemetry unavailable", exc_info=True)

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
            "litellm_telemetry": litellm_telemetry,
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
    role_aliases = registry.role_model_aliases()
    role_bindings = registry.role_model_bindings()
    worker_aliases = {
        str(entry.get("public_model_alias")): entry
        for entry in registry.native_worker_manifest().values()
        if entry.get("public_model_alias")
    }
    # Slot aliases are profile-scoped at request time. Advertise the
    # operator's primary profile projection in the catalog while retaining
    # ordinary logical models and role aliases for compatibility.
    slot_profile_id = (
        "freeinference" if "freeinference" in registry.profiles
        else next(iter(registry.profiles), None)
    )
    slot_entries = registry.slot_alias_manifest(slot_profile_id)
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
            "display_name": (
                f"Brigade {alias.removeprefix('anthropic-brigade-').replace('-', ' ').title()}"
            ),
            "type": "model",
            "alias_kind": (
                "sidecar_agent"
                if worker_aliases.get(alias, {}).get("source_kind") == "sidecar_agent"
                else "role"
            ),
            "role": role,
            **(
                {
                    "native_agent_name": worker_aliases[alias]["native_agent_name"],
                    "worker_id": (
                        worker_aliases[alias].get("worker_id")
                        or worker_aliases[alias].get("sidecar_agent_id")
                    ),
                    "can_mutate": bool(worker_aliases[alias].get("can_mutate")),
                    "tools": worker_aliases[alias].get("tools", []),
                }
                if alias in worker_aliases
                and worker_aliases[alias].get("source_kind") == "sidecar_agent"
                else {}
            ),
            "provider": (
                registry.models[role_bindings[alias]].provider_id
                if alias in role_bindings and role_bindings[alias] in registry.models
                else None
            ),
            "controller_eligible": False,
            "status": "configured",
            **(
                {"backing_model": role_bindings[alias]}
                if alias in role_bindings
                else {}
            ),
        }
        for alias, role in sorted(role_aliases.items())
    )
    entries.extend(
        {
            "id": entry["public_model_alias"],
            "display_name": f"Brigade Slot {slot.title()}",
            "type": "model",
            "alias_kind": "slot",
            "slot": slot,
            "provider": (
                registry.models[entry["model_id"]].provider_id
                if entry["model_id"] in registry.models else entry.get("provider_id")
            ),
            "controller_eligible": slot == "main",
            "status": "configured",
            "backing_model": entry["model_id"],
            "model_alias": entry["model_alias"],
        }
        for slot, entry in sorted(slot_entries.items())
        if entry["public_model_alias"] not in role_aliases
    )
    return {"data": entries, "has_more": False}


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    _require_local(request)
    payload = await _read_json_request(request)
    if not isinstance(payload.get("model"), str):
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
    from enhanced_router.registry import get_registry

    payload = await _read_json_request(request)
    if not isinstance(payload.get("model"), str):
        raise HTTPException(status_code=400, detail="request requires a model")
    model = payload["model"]

    # Role aliases — local estimation only
    if model in get_registry().role_model_aliases():
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
