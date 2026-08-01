"""Backend dispatch interfaces and helper utilities."""

from __future__ import annotations

from enhanced_router.agent_manifest import GENERIC_ROLE_ALIASES
from enhanced_router.base import HOP_BY_HOP
from enhanced_router.config_models import ModelAuthSpec

import json
import asyncio
from datetime import timezone
from email.utils import parsedate_to_datetime
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel
import httpx
from enhanced_router.provider_admission import AdmissionTimeout, ProviderAdmissionManager, ProviderLimits

# ---------------------------------------------------------------------------
# Typed auth spec for provider credential resolution
# ---------------------------------------------------------------------------

class ProviderAuthSpec(BaseModel):
    type: Literal["bearer", "x-api-key", "custom-header", "none"]
    env: str | None = None
    header_name: str | None = None


def _credential_value(key_name: str | None) -> str | None:
    """Resolve a router-local rotating credential without exposing it upstream."""
    if not key_name:
        return None
    try:
        from enhanced_router.credential_store import resolve_loaded

        value = resolve_loaded(key_name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(key_name) or None


def resolve_provider_auth(
    api_key_env: str | None,
    auth_spec: ModelAuthSpec | None = None,
) -> ProviderAuthSpec:
    """Resolve provider-specific auth configuration from the model's explicit auth spec.

    P0-2: No longer infers auth from backend type. Uses the model's validated
    ModelAuthSpec which is pinned at binding time.

    Parameters
    ----------
    api_key_env :
        The environment variable name holding the provider credential.
    auth_spec :
        The model's explicit auth specification (ModelAuthSpec). If None and
        no api_key_env, returns "none" type.

    Returns
    -------
    ProviderAuthSpec
        Auth resolution with env, header, and type.
    """
    if not api_key_env:
        return ProviderAuthSpec(type="none")
    if auth_spec is None:
        # Fallback for legacy bindings without pinned auth spec
        # This path should not be taken for new bindings
        return ProviderAuthSpec(type="bearer", env=api_key_env, header_name="authorization")
    # Use the model's explicit auth specification
    return ProviderAuthSpec(
        type=auth_spec.type,
        env=api_key_env,
        header_name=auth_spec.header,
    )


def _resolve_pinned_auth(
    auth_spec_json: str | None,
    api_key_env: str | None,
    backend: str,
) -> tuple[str, str]:
    """Resolve the credential header name and formatted value from the pinned auth spec.

    P0-2: Uses the model's validated auth specification (pinned in the binding)
    rather than inferring auth from the backend category.

    Returns (header_name, header_value) ready for injection, or raises ValueError
    if the credential is unavailable.
    """
    if auth_spec_json:
        try:
            spec = ModelAuthSpec(**json.loads(auth_spec_json))
            env = api_key_env
            if not env:
                raise ValueError("No api_key_env in binding for credential lookup")
            key = _credential_value(env)
            if not key:
                raise ValueError(f"API key env var '{env}' is not set")
            header = spec.effective_header()
            value = spec.format_value(key)
            return (header, value)
        except ValueError:
            raise
        except Exception:
            pass  # fall through to legacy resolution

    # Legacy fallback for bindings without pinned auth_spec_json
    auth = resolve_provider_auth(api_key_env, None)
    if auth.type == "none" or not auth.env:
        raise ValueError("No credential configured")
    key = _credential_value(auth.env)
    if not key:
        raise ValueError(f"API key env var '{auth.env}' is not set")
    if auth.header_name == "x-api-key":
        return ("x-api-key", key)
    elif auth.header_name == "authorization":
        return ("authorization", f"Bearer {key}")
    elif auth.header_name:
        return (auth.header_name, key)
    else:
        return ("authorization", f"Bearer {key}")

LOGGER = logging.getLogger("claude-enhanced-router")
_provider_admission = ProviderAdmissionManager()
_litellm_supervisor: Any | None = None
_ANTHROPIC_PASSTHROUGH_PROVIDER = "anthropic"


def configure_litellm_supervisor(supervisor: Any | None) -> None:
    """Attach the process supervisor used for generation activity accounting."""
    global _litellm_supervisor
    _litellm_supervisor = supervisor


def configure_provider_admission(limits: dict[str, ProviderLimits]) -> None:
    """Apply registry-owned provider limits to the shared transport manager."""
    # Claude subscription passthrough does not have a registry model/provider
    # row, but it still needs the same bounded request and stream lifecycle as
    # every other outbound backend.  Operators may override this synthetic
    # lane by declaring an explicit ``anthropic`` provider.
    configured = dict(limits)
    configured.setdefault(
        _ANTHROPIC_PASSTHROUGH_PROVIDER,
        ProviderLimits(max_concurrency=8, max_queued_agents=0, queue_timeout_seconds=20.0),
    )
    for provider_id, provider_limits in configured.items():
        _provider_admission.configure(provider_id, provider_limits)


def provider_admission_snapshots() -> dict[str, dict[str, object]]:
    """Return provider-wide active/queued counts for health and diagnostics."""
    return _provider_admission.snapshots()


class BackendType(Enum):
    """Supported backend kinds."""

    ANTHROPIC_PASSTHROUGH = "anthropic_passthrough"
    DIRECT_ANTHROPIC = "direct_anthropic"
    LITELLM = "litellm"


class UnsupportedBackendError(RuntimeError):
    """Raised when a binding contains an unknown backend value."""

    def __init__(self, backend: str) -> None:
        super().__init__(
            f"Unsupported backend '{backend}'. "
            "Known backends: direct-anthropic, litellm, anthropic-passthrough."
        )
        self.backend = backend


@dataclass(frozen=True)
class RequestIdentity:
    """Complete Claude identity tuple for a request."""

    run_id: str | None
    claude_session_id: str | None
    claude_agent_id: str | None
    claude_parent_agent_id: str | None
    endpoint_kind: str


@dataclass(frozen=True)
class ResolvedRoute:
    """Resolved routing decision for an incoming request."""

    kind: BackendType
    role: str | None = None
    model_id: str | None = None
    upstream_model: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    agent_binding_id: int | None = None
    route_version: int | None = None
    registry_hash: str | None = None
    catalog_generation: int | None = None
    litellm_model_name: str | None = None
    litellm_base_url: str | None = None
    auth_spec_json: str | None = None
    provider_id: str | None = None
    endpoint_id: str | None = None
    endpoint_selection_reason: str | None = None
    routing_mode: str = "fixed"
    deployment_group: str | None = None
    allowed_deployments: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    deployment_policy_digest: str | None = None
    controller_binding_id: int | None = None


# Compatibility exports for callers that only need the four stable role
# aliases.  Model-qualified identities are registry-owned; routing and
# scheduling obtain them from ``ModelRegistry.role_model_aliases()`` and
# ``ModelRegistry.role_model_bindings()``.
ROLE_MODEL_ALIASES: dict[str, str] = dict(GENERIC_ROLE_ALIASES)
ROLE_MODEL_BINDINGS: dict[str, str] = {}


def parse_request_identity(request: Any) -> RequestIdentity:
    """Extract the complete Claude identity tuple from request headers.

    Reads Brigade-specific headers and Claude Code identity headers to
    build a ``RequestIdentity`` suitable for identity-first routing.

    Parameters
    ----------
    request :
        The incoming FastAPI request.

    Returns
    -------
    RequestIdentity
        Identity tuple with values from headers or ``None`` when absent.
    """
    return RequestIdentity(
        run_id=request.headers.get("x-brigade-run-id"),
        claude_session_id=request.headers.get("x-claude-code-session-id"),
        claude_agent_id=request.headers.get("x-claude-code-agent-id"),
        claude_parent_agent_id=request.headers.get("x-claude-code-parent-agent-id"),
        endpoint_kind=request.url.path,
    )

# Claude OAuth and auth headers — stripped for all non-anthropic backends
_CLAUDE_OAUTH_HEADERS = frozenset({
    "authorization",
    "x-api-key",
})

# Internal Brigade headers
_BRIGADE_HEADERS = frozenset({
    "x-enhanced-token",
    "x-brigade-run-id",
    "x-brigade-session-id",
})

# x-claude-code-* headers are internal to Claude Code; never forward to any backend
_CLAUDE_CODE_PREFIX = "x-claude-code-"

# Anthropic-internal headers; strip everything except the standard API version
_ANTHROPIC_HEADERS_ALLOWED = frozenset({"anthropic-version"})


class SanitizeMode(str, Enum):
    """Header sanitization policy per backend kind.

    Each mode defines which headers are preserved, stripped, or injected.
    """
    ANTHROPIC_PASSTHROUGH = "anthropic_passthrough"
    EXTERNAL_PROVIDER = "external_provider"
    LITELLM_INTERNAL = "litellm_internal"
    DIRECT_ANTHROPIC = "direct_anthropic"


def sanitize_headers_for_backend(
    headers: dict[str, str],
    mode: SanitizeMode,
    *,
    inject_auth_header: str | None = None,
    inject_auth_value: str | None = None,
) -> dict[str, str]:
    """Sanitize request headers for the given backend mode.

    Parameters
    ----------
    headers :
        Raw incoming request headers.
    mode :
        Sanitization policy — controls which headers are stripped/injected.
    inject_auth_header :
        Header name for optional credential injection.
    inject_auth_value :
        Header value for optional credential injection.

    Returns
    -------
    dict[str, str]
        Cleaned headers ready for upstream forwarding.
    """
    cleaned: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()

        # Always strip HOP-BY-HOP headers (RFC 9113 §8.2.2)
        if lower in HOP_BY_HOP:
            continue

        # Claude OAuth headers stripped for all non-passthrough modes
        if mode != SanitizeMode.ANTHROPIC_PASSTHROUGH:
            if lower in _CLAUDE_OAUTH_HEADERS:
                continue
            if lower == "x-api-key":
                continue

        # Never forward x-claude-code-* headers to any backend
        if lower.startswith(_CLAUDE_CODE_PREFIX):
            continue

        # Internal Brigade headers — always strip
        if lower in _BRIGADE_HEADERS:
            continue

        # Anthropic-internal headers (except standard version header)
        if lower.startswith("anthropic-"):
            if mode == SanitizeMode.ANTHROPIC_PASSTHROUGH:
                # Passthrough preserves most anthropic-* headers for Claude
                cleaned[key] = value
                continue
            if lower in _ANTHROPIC_HEADERS_ALLOWED:
                cleaned[key] = value
            continue

        # Preserve all other standard headers
        cleaned[key] = value

    # Inject content-type for non-passthrough modes (passthrough lets
    # the caller set it explicitly to preserve existing behavior).
    if mode != SanitizeMode.ANTHROPIC_PASSTHROUGH:
        cleaned.setdefault("content-type", "application/json")

    # Inject auth credentials if provided
    if inject_auth_header and inject_auth_value:
        cleaned[inject_auth_header] = inject_auth_value

    return cleaned


def sanitize_upstream_headers(headers: dict[str, str]) -> dict[str, str]:
    """Legacy wrapper — delegates to ``sanitize_headers_for_backend`` with
    ``ANTHROPIC_PASSTHROUGH`` mode for backward compatibility."""
    return sanitize_headers_for_backend(headers, SanitizeMode.ANTHROPIC_PASSTHROUGH)


def _remove_local_router_credential(headers: dict[str, str]) -> None:
    """Prevent the loopback session token from crossing an upstream boundary."""
    token = os.environ.get("ENHANCED_ROUTER_TOKEN", "")
    if not token:
        return
    if headers.get("authorization", "") == f"Bearer {token}":
        headers.pop("authorization", None)
    if headers.get("x-api-key", "") == token:
        headers.pop("x-api-key", None)


# ---------------------------------------------------------------------------
# Proxy implementations
# ---------------------------------------------------------------------------

# Shared external upstream HTTP client (lazily initialized per event loop).
_client: Any = None


def get_upstream_client() -> Any:
    global _client
    if _client is None:
        import httpx

        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0, read=90.0, write=120.0, pool=30.0
            ),
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
    return _client


async def close_upstream_client() -> None:
    """Close the module-level HTTPX client if it was created."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None



def copy_response_headers(headers: Any) -> dict[str, str]:
    """Strip hop-by-hop headers from upstream response headers."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP
    }


async def _stream_upstream(response: Any) -> AsyncIterator[bytes]:
    """Async generator yielding bytes from an httpx response."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def _consume_sse_usage_lines(
    buffer: bytearray,
    chunk: bytes,
    current_usage: dict[str, Any] | None,
    *,
    max_line_bytes: int = 131_072,
) -> tuple[dict[str, Any] | None, bool]:
    """Consume complete SSE lines without rescanning previously parsed data."""
    buffer.extend(chunk)
    overflowed = False
    while True:
        newline = buffer.find(b"\n")
        if newline < 0:
            break
        line = bytes(buffer[:newline]).rstrip(b"\r")
        del buffer[: newline + 1]
        if not line.startswith(b"data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == b"[DONE]":
            continue
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        usage_sources = [event.get("usage")]
        for container_name in ("message", "delta"):
            container = event.get(container_name)
            if isinstance(container, dict):
                usage_sources.append(container.get("usage"))
        for usage in usage_sources:
            if not isinstance(usage, dict):
                continue
            # Anthropic message_start/message_delta and OpenAI-compatible
            # usage events can split input/output/cache fields across events.
            # Keep a monotonic accumulator so a later partial event cannot
            # erase evidence observed earlier in the stream.
            merged: dict[str, Any] = {}
            if isinstance(current_usage, dict):
                prior = current_usage.get("usage")
                if isinstance(prior, dict):
                    merged.update(prior)
            usage_complete = bool(
                current_usage.get("usage_complete", False)
                if isinstance(current_usage, dict) else False
            )
            for key, value in usage.items():
                if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                    merged[key] = max(merged[key], value)
                else:
                    merged[key] = value
            usage_complete = usage_complete or event.get("type") in {
                "message_delta", "message_stop", "response.completed",
            } or (event.get("choices") == [] and "usage" in event)
            current_usage = (
                {"usage": merged}
                if usage_complete
                else {"usage": merged, "usage_complete": False}
            )

    if len(buffer) > max_line_bytes:
        buffer.clear()
        overflowed = True
    return current_usage, overflowed


async def _stream_upstream_with_admission(
    response: Any,
    request_id: str,
    resolved: ResolvedRoute | None = None,
    started_at: float | None = None,
) -> AsyncIterator[bytes]:
    """Release a provider request slot only after the full stream closes."""
    pending = bytearray()
    usage_payload: dict[str, Any] | None = None
    usage_line_overflow = False
    succeeded = 200 <= int(getattr(response, "status_code", 500)) < 300
    ttft_seconds, idle_seconds, wall_seconds = _provider_stream_deadlines(resolved)
    iterator = response.aiter_bytes().__aiter__()
    saw_token = False
    try:
        while True:
            elapsed = time.perf_counter() - started_at if started_at is not None else 0.0
            remaining = wall_seconds - elapsed
            if remaining <= 0:
                raise TimeoutError("provider request wall deadline exceeded")
            wait_seconds = min(ttft_seconds if not saw_token else idle_seconds, remaining)
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=wait_seconds)
            except StopAsyncIteration:
                break
            if chunk:
                saw_token = True
            if resolved is not None:
                usage_payload, overflowed = _consume_sse_usage_lines(
                    pending, chunk, usage_payload
                )
                usage_line_overflow |= overflowed
            yield chunk
    except asyncio.CancelledError:
        succeeded = False
        if resolved is not None:
            _record_execution_transport_failure(
                resolved, "cancelled_by_controller", "stream cancelled",
            )
        raise
    except (TimeoutError, asyncio.TimeoutError) as exc:
        succeeded = False
        if resolved is not None:
            _record_execution_transport_failure(
                resolved,
                "stream_idle_timeout" if saw_token else "ttft_timeout",
                str(exc),
            )
        raise
    except Exception as exc:
        succeeded = False
        if resolved is not None:
            _record_execution_transport_failure(
                resolved, "malformed_stream", str(exc),
            )
        raise
    finally:
        await response.aclose()
        if resolved is not None and pending:
            usage_payload, overflowed = _consume_sse_usage_lines(
                pending, b"\n", usage_payload
            )
            usage_line_overflow |= overflowed
        if usage_line_overflow:
            LOGGER.warning("request=%s contained an oversized SSE usage line", request_id)
        if resolved is not None and usage_payload is not None:
            usage_route = _route_with_reported_deployment(resolved, response.headers)
            _record_endpoint_usage(
                usage_route,
                request_id,
                usage_payload,
                started_at,
                succeeded=succeeded,
            )
        await _release_provider_request(resolved, request_id)


def _record_execution_transport_failure(
    resolved: ResolvedRoute,
    error_class: str,
    reason: str,
) -> None:
    if resolved.agent_binding_id is None:
        return
    try:
        from enhanced_router.state import get_state
        get_state().record_execution_failure_for_binding(
            resolved.agent_binding_id, error_class=error_class, error=reason,
        )
    except Exception:
        LOGGER.debug("failed to persist stream failure", exc_info=True)


def _provider_stream_deadlines(resolved: "ResolvedRoute | None") -> tuple[float, float, float]:
    """Return provider-owned TTFT, idle-stream and wall deadlines."""
    defaults = (45.0, 90.0, 600.0)
    if resolved is None:
        return defaults
    try:
        from enhanced_router.registry import get_registry

        providers = get_registry().providers
        provider_ids = _provider_ids_for_route(resolved)
        selected = [providers[provider_id] for provider_id in provider_ids if provider_id in providers]
        if not selected:
            return defaults
        # A managed group must remain bounded even when LiteLLM changes the
        # physical deployment.  Use the strictest candidate deadline.
        return (
            min(provider.deadlines.time_to_first_token_seconds for provider in selected),
            min(provider.deadlines.stream_idle_seconds for provider in selected),
            min(provider.deadlines.request_wall_seconds for provider in selected),
        )
    except (ImportError, KeyError, AttributeError, RuntimeError):
        return defaults


async def _send_with_deadline(
    client: Any,
    request: Any,
    resolved: "ResolvedRoute",
    started_at: float,
) -> Any:
    """Bound response headers by TTFT and the absolute request budget.

    Waiting for response headers is part of time-to-first-token for streaming
    requests.  Using only the wall deadline here allowed an upstream to hold
    a connection open for minutes before the body timeout was ever reached.
    """
    ttft_seconds, _, wall_seconds = _provider_stream_deadlines(resolved)
    remaining = max(0.001, wall_seconds - (time.perf_counter() - started_at))
    return await asyncio.wait_for(
        client.send(request, stream=True), timeout=min(ttft_seconds, remaining)
    )


async def _read_with_deadline(
    response: Any,
    resolved: "ResolvedRoute",
    started_at: float,
) -> bytes:
    """Bound non-stream response body reads by the same request budget."""
    _, _, wall_seconds = _provider_stream_deadlines(resolved)
    remaining = max(0.001, wall_seconds - (time.perf_counter() - started_at))
    return await asyncio.wait_for(response.aread(), timeout=remaining)


async def _send_with_provider_retry(
    client: Any,
    request_builder: Any,
    resolved: "ResolvedRoute",
    started_at: float,
    request_id: str | None,
) -> Any:
    """Send one pinned request with bounded provider-configured retries.

    Retries never change model, endpoint, binding, or request identity.  The
    provider request permit remains held across the complete retry sequence.
    """
    max_attempts = 1
    retryable: set[int] = set()
    max_backoff = 15.0
    try:
        from enhanced_router.registry import get_registry

        provider = get_registry().providers.get(resolved.provider_id or "")
        if provider is not None:
            max_attempts = max(1, provider.retry.max_attempts)
            retryable = set(provider.retry.retryable_statuses)
            max_backoff = provider.retry.max_backoff_seconds
    except (ImportError, AttributeError, RuntimeError):
        pass

    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            response = await _send_with_deadline(
                client, request_builder(), resolved, started_at
            )
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                raise
            if not await _provider_admission.retry_allowed(
                _provider_ids_for_route(resolved), request_id or "retry",
            ):
                raise AdmissionTimeout("provider circuit opened during retry")
            ceiling = min(max_backoff, 0.5 * (2**attempt))
            delay = random.uniform(0.0, max(0.1, ceiling))
            _, _, wall_seconds = _provider_stream_deadlines(resolved)
            if time.perf_counter() - started_at + delay >= wall_seconds:
                raise TimeoutError("provider retry budget exceeded request wall deadline")
            await asyncio.sleep(delay)
            continue

        status = int(getattr(response, "status_code", 500))
        if status in retryable and attempt + 1 < max_attempts:
            await _record_provider_response(resolved, request_id, response)
            retry_after = _retry_after_seconds(getattr(response, "headers", {}).get("retry-after"))
            await response.aclose()
            if not await _provider_admission.retry_allowed(
                _provider_ids_for_route(resolved), request_id or "retry",
            ):
                raise AdmissionTimeout("provider circuit opened during retry")
            ceiling = min(max_backoff, retry_after or 0.5 * (2**attempt))
            delay = random.uniform(0.0, max(0.1, ceiling))
            _, _, wall_seconds = _provider_stream_deadlines(resolved)
            if time.perf_counter() - started_at + delay >= wall_seconds:
                raise TimeoutError("provider retry budget exceeded request wall deadline")
            await asyncio.sleep(delay)
            continue
        await _record_provider_response(resolved, request_id, response)
        return response

    raise RuntimeError("provider request exhausted its retry budget") from last_error


async def post_openai_compatible_json(
    *,
    api_base: str,
    model: str,
    api_key_env: str,
    provider_id: str,
    endpoint_id: str | None,
    payload: dict[str, Any],
    request_id: str,
    extra_headers: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
    priority: bool = False,
) -> dict[str, Any]:
    """Send one bounded non-streaming OpenAI-compatible request.

    This is the protocol adapter for router-owned JSON specialists such as
    fastpath. Admission, provider retry/circuit checks, absolute deadlines,
    response accounting, and permit release remain shared with normal model
    backends.
    """
    key = _credential_value(api_key_env)
    if not key:
        raise RuntimeError(f"credential '{api_key_env}' is unavailable")
    route = ResolvedRoute(
        kind=BackendType.LITELLM,
        model_id=model,
        api_base=api_base.rstrip("/"),
        api_key_env=api_key_env,
        provider_id=provider_id,
        endpoint_id=endpoint_id or "openai-chat",
    )
    headers = {
        "authorization": f"Bearer {key}",
        "content-type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps({**payload, "model": model}, separators=(",", ":")).encode("utf-8")
    request = httpx.Request(
        "POST", f"{route.api_base}/chat/completions", headers=headers, content=body,
    )
    started_at = time.perf_counter()
    admission_id = await _acquire_provider_request(route, request, streaming=False, priority=priority)
    if admission_id is None:
        raise RuntimeError("OpenAI-compatible request admission did not return a request ID")
    client = get_upstream_client()
    upstream_response: Any | None = None
    try:
        send = _send_with_provider_retry(
            client,
            lambda: client.build_request(
                request.method,
                str(request.url),
                headers=dict(request.headers),
                content=body,
            ),
            route,
            started_at,
            admission_id,
        )
        if timeout_seconds is not None:
            upstream_response = await asyncio.wait_for(
                send,
                timeout=max(0.001, timeout_seconds - (time.perf_counter() - started_at)),
            )
        else:
            upstream_response = await send
        if upstream_response is None:
            raise RuntimeError("OpenAI-compatible upstream returned no response")
        read = _read_with_deadline(upstream_response, route, started_at)
        if timeout_seconds is not None:
            content = await asyncio.wait_for(
                read,
                timeout=max(0.001, timeout_seconds - (time.perf_counter() - started_at)),
            )
        else:
            content = await read
        upstream_response.raise_for_status()
    finally:
        try:
            if upstream_response is not None:
                await upstream_response.aclose()
        finally:
            await _release_provider_request(route, admission_id)
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("OpenAI-compatible response was malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI-compatible response was not an object")
    _record_endpoint_usage(
        route,
        admission_id,
        parsed,
        started_at,
        succeeded=True,
    )
    return parsed


def _retry_after_seconds(value: object) -> float | None:
    """Parse both RFC 7231 delta-seconds and HTTP-date Retry-After values."""
    if value is None:
        return None
    raw = str(value).strip()
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(raw)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0.0, date.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _usage_totals(payload: dict[str, Any]) -> dict[str, int] | None:
    """Normalize common OpenAI/Anthropic usage shapes without retaining content."""
    usage = payload.get("usage", payload)
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = {}
    cache_read = int(
        usage.get("cache_read_input_tokens", details.get("cached_tokens", 0)) or 0
    )
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = int(usage.get("input_tokens", 0) or 0) + cache_read + cache_write
    return {
        "input_tokens": max(0, int(input_tokens or 0)),
        "cache_read_tokens": max(0, cache_read),
        "cache_write_tokens": max(0, cache_write),
        "output_tokens": max(
            0,
            int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
        ),
    }


def _record_endpoint_usage(
    resolved: "ResolvedRoute",
    request_id: str | None,
    payload: dict[str, Any],
    started_at: float | None,
    *,
    succeeded: bool,
) -> None:
    """Persist sanitized token/cache metrics for a logical model endpoint."""
    if not resolved.provider_id or not resolved.model_id or not resolved.endpoint_id:
        return
    if payload.get("usage_complete") is False:
        LOGGER.debug(
            "ignoring incomplete stream usage model=%s endpoint=%s",
            resolved.model_id, resolved.endpoint_id,
        )
        return
    totals = _usage_totals(payload)
    if totals is None:
        return
    try:
        from enhanced_router.state import get_state

        get_state().record_endpoint_usage(
            provider_id=resolved.provider_id,
            model_id=resolved.model_id,
            endpoint_id=resolved.endpoint_id,
            request_id=request_id,
            input_tokens=totals["input_tokens"],
            cache_read_tokens=totals["cache_read_tokens"],
            cache_write_tokens=totals["cache_write_tokens"],
            output_tokens=totals["output_tokens"],
            latency_ms=(time.perf_counter() - started_at) * 1000 if started_at else None,
            succeeded=succeeded,
            configuration_hash=resolved.registry_hash or "",
        )
        get_state().record_execution_metrics_for_binding(
            resolved.agent_binding_id,
            input_tokens=totals["input_tokens"],
            cache_read_tokens=totals["cache_read_tokens"],
            cache_write_tokens=totals["cache_write_tokens"],
            output_tokens=totals["output_tokens"],
            wall_time_ms=(time.perf_counter() - started_at) * 1000 if started_at else None,
        )
    except Exception:
        LOGGER.exception(
            "failed to record endpoint usage model=%s endpoint=%s",
            resolved.model_id,
            resolved.endpoint_id,
        )


def _route_with_reported_deployment(
    resolved: "ResolvedRoute",
    headers: Any,
) -> "ResolvedRoute":
    """Attach LiteLLM's reported deployment to a managed-group observation."""
    if resolved.endpoint_id or not resolved.allowed_deployments:
        return resolved
    reported = (
        headers.get("x-litellm-deployment-id")
        or headers.get("x-litellm-model-id")
        or headers.get("x-litellm-model")
    )
    if not reported:
        return resolved
    reported = str(reported)
    endpoint_id = reported if reported in set(resolved.allowed_deployments) else None
    if endpoint_id is None:
        LOGGER.warning(
            "managed deployment identity is unknown or ambiguous model=%s reported=%s",
            resolved.model_id, reported,
        )
        return resolved
    provider_id = resolved.provider_id
    try:
        from enhanced_router.registry import get_registry

        spec = get_registry().get_model(str(resolved.model_id))
        endpoint = spec.endpoints.get(endpoint_id)
        if endpoint is not None:
            provider_id = endpoint.provider_id or spec.provider_id
    except (KeyError, AttributeError, RuntimeError):
        pass
    return replace(resolved, endpoint_id=endpoint_id, provider_id=provider_id)


async def _acquire_provider_request(
    resolved: "ResolvedRoute", request: Any, *, streaming: bool = False, priority: bool = False,
) -> str | None:
    provider_ids = _provider_ids_for_route(resolved)
    if not provider_ids and resolved.catalog_generation is None:
        return None
    # Keep the caller's ID for correlation, but always add a unique suffix so
    # a reused/malformed upstream ID can never bypass admission as a duplicate.
    caller_request_id = request.headers.get("x-request-id") or "request"
    request_id = f"{caller_request_id}:{uuid.uuid4().hex}"
    try:
        from enhanced_router.state import get_state
        get_state().increment_execution_requests(resolved.agent_binding_id)
    except Exception:
        LOGGER.debug("request could not be correlated to an execution", exc_info=True)
    try:
        if len(provider_ids) > 1:
            # Managed provider groups don't support priority admission --
            # fastpath (the only current priority caller) never resolves to
            # a multi-provider group.
            await _provider_admission.acquire_request_group(provider_ids, request_id)
        elif provider_ids:
            await _provider_admission.acquire_request(
                next(iter(provider_ids)), request_id, priority=priority,
            )
    except AdmissionTimeout as exc:
        raise RuntimeError(f"provider request admission timed out: {exc}") from exc
    if resolved.catalog_generation is not None and _litellm_supervisor is not None:
        _litellm_supervisor.track_request_started(
            resolved.catalog_generation,
            request_id,
            streaming=streaming,
        )
    return request_id


def _provider_ids_for_route(resolved: "ResolvedRoute") -> tuple[str, ...]:
    """Return the immutable provider candidate set for this route."""
    if resolved.provider_ids:
        return tuple(sorted(set(resolved.provider_ids)))
    if resolved.provider_id:
        return (resolved.provider_id,)
    return ()


def _passthrough_admission_route(resolved: "ResolvedRoute") -> "ResolvedRoute":
    """Attach the synthetic Anthropic provider identity to passthrough routes."""
    if resolved.provider_id or resolved.provider_ids:
        return resolved
    return replace(
        resolved,
        provider_id=_ANTHROPIC_PASSTHROUGH_PROVIDER,
        endpoint_id=resolved.endpoint_id or "messages",
    )


async def _release_provider_request(
    resolved: "ResolvedRoute | None", request_id: str | None,
) -> None:
    if not request_id or resolved is None:
        return
    if resolved.catalog_generation is not None and _litellm_supervisor is not None:
        _litellm_supervisor.track_request_finished(
            resolved.catalog_generation, request_id,
        )
    provider_ids = _provider_ids_for_route(resolved)
    if len(provider_ids) > 1:
        await _provider_admission.release_request_group(provider_ids, request_id)
    elif provider_ids:
        await _provider_admission.release_request(request_id)


async def _record_provider_response(
    resolved: "ResolvedRoute", request_id: str | None, response: Any,
) -> None:
    """Feed response-level rate/unavailability state into shared admission."""
    if not request_id:
        return
    reported = _route_with_reported_deployment(resolved, getattr(response, "headers", {}))
    provider_id = reported.provider_id
    # With several managed providers and no trusted deployment identity, do
    # not attribute a failure to every provider.  The request reservation is
    # still released conservatively from all candidates.
    if not provider_id:
        return
    retry_after: float | None = None
    raw_retry_after = getattr(response, "headers", {}).get("retry-after")
    if raw_retry_after:
        retry_after = _retry_after_seconds(raw_retry_after)
    await _provider_admission.record_response(
        provider_id,
        request_id,
        int(getattr(response, "status_code", 500)),
        retry_after_seconds=retry_after,
    )


async def proxy_anthropic_passthrough(
    request: Any, payload: dict[str, Any]
) -> Any:
    """Forward the request directly to Anthropic upstream.

    Used for standard Claude model IDs that are not role aliases.
    This is the fallback path -- the main Claude subscription passes through
    unchanged.
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    upstream = "https://api.anthropic.com"
    path = request.url.path
    query = request.url.query
    url = f"{upstream}{path}" + (f"?{query}" if query else "")

    raw_headers = dict(request.headers)
    headers = sanitize_headers_for_backend(raw_headers, SanitizeMode.ANTHROPIC_PASSTHROUGH)
    _remove_local_router_credential(headers)
    headers["content-type"] = "application/json"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    is_streaming = bool(payload.get("stream"))
    resolved = ResolvedRoute(
        kind=BackendType.ANTHROPIC_PASSTHROUGH,
        model_id=str(payload.get("model") or "anthropic"),
    )
    admission_route = _passthrough_admission_route(resolved)
    started_at = time.perf_counter()
    admission_id: str | None = None

    client = get_upstream_client()
    try:
        admission_id = await _acquire_provider_request(
            admission_route, request, streaming=is_streaming,
        )
        if admission_id is None:
            raise RuntimeError("Anthropic passthrough admission did not return a request ID")
        upstream_response = await _send_with_provider_retry(
            client,
            lambda: client.build_request(
                request.method, url, headers=headers, content=body
            ),
            admission_route,
            started_at,
            admission_id,
        )
    except Exception as exc:
        await _release_provider_request(admission_route, admission_id)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_connection_error",
                    "message": str(exc),
                }
            },
        )

    response_headers = copy_response_headers(upstream_response.headers)
    if is_streaming:
        return StreamingResponse(
            _stream_upstream_with_admission(
                upstream_response, admission_id, admission_route, started_at
            ),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await _read_with_deadline(upstream_response, admission_route, started_at)
    finally:
        await upstream_response.aclose()
        await _release_provider_request(admission_route, admission_id)
    try:
        response_payload = json.loads(content)
    except (TypeError, ValueError):
        response_payload = None
    if isinstance(response_payload, dict):
        _record_endpoint_usage(
            admission_route,
            admission_id,
            response_payload,
            started_at,
            succeeded=200 <= upstream_response.status_code < 300,
        )
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def proxy_anthropic_passthrough_worker(
    request: Any,
    payload: dict[str, Any],
    resolved: "ResolvedRoute",
) -> Any:
    """Forward a role-bound worker request to Anthropic upstream.

    Used when a role alias resolves to ``anthropic-passthrough`` backend.
    Preserves Claude OAuth headers (does not strip authorization),
    rewrites the model to the pinned upstream model, and forwards
    directly to ``https://api.anthropic.com``.

    Parameters
    ----------
    request :
        Incoming FastAPI request.
    payload :
        Normalised Anthropic messages payload.
    resolved :
        The resolved route with pinned upstream model and binding metadata.
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    upstream = "https://api.anthropic.com"
    path = request.url.path
    query = request.url.query
    url = f"{upstream}{path}" + (f"?{query}" if query else "")

    # Rewrite model to the pinned upstream Claude model
    payload["model"] = resolved.upstream_model or payload.get("model")

    # Build headers: preserve Claude OAuth headers (authorization),
    # strip only internal Brigade headers.
    raw_headers = dict(request.headers)
    headers = sanitize_upstream_headers(raw_headers)
    _remove_local_router_credential(headers)
    headers["content-type"] = "application/json"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    is_streaming = bool(payload.get("stream"))
    started_at = time.perf_counter()
    admission_route = _passthrough_admission_route(resolved)
    admission_id: str | None = None

    LOGGER.info(
        "anthropic_passthrough_worker role=%s model=%s session=%s agent=%s binding=%s",
        resolved.role,
        payload.get("model"),
        request.headers.get("x-claude-code-session-id", "-"),
        request.headers.get("x-claude-code-agent-id", "main"),
        resolved.agent_binding_id,
    )

    client = get_upstream_client()
    try:
        admission_id = await _acquire_provider_request(
            admission_route, request, streaming=is_streaming,
        )
        if admission_id is None:
            raise RuntimeError("Anthropic passthrough admission did not return a request ID")
        upstream_response = await _send_with_provider_retry(
            client,
            lambda: client.build_request(
                request.method, url, headers=headers, content=body
            ),
            admission_route,
            started_at,
            admission_id,
        )
    except Exception as exc:
        await _release_provider_request(admission_route, admission_id)
        LOGGER.warning(
            "anthropic_passthrough_worker error role=%s error=%s",
            resolved.role, exc,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_connection_error",
                    "message": str(exc),
                }
            },
        )

    response_headers = copy_response_headers(upstream_response.headers)
    if is_streaming:
        return StreamingResponse(
            _stream_upstream_with_admission(
                upstream_response, admission_id, admission_route, started_at
            ),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await _read_with_deadline(upstream_response, admission_route, started_at)
    finally:
        await upstream_response.aclose()
        await _release_provider_request(admission_route, admission_id)
    try:
        response_payload = json.loads(content)
    except (TypeError, ValueError):
        response_payload = None
    if isinstance(response_payload, dict):
        _record_endpoint_usage(
            _route_with_reported_deployment(admission_route, upstream_response.headers),
            admission_id,
            response_payload,
            started_at,
            succeeded=200 <= upstream_response.status_code < 300,
        )
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def proxy_direct_anthropic(
    request: Any,
    payload: dict[str, Any],
    resolved: "ResolvedRoute",
) -> Any:
    """Route worker traffic to a direct Anthropic-compatible backend (e.g. LongCat).

    Security contract (P0-5):

    - Strips ALL Claude OAuth / authorization headers from the incoming request.
    - Strips x-api-key, anthropic-beta (unless overridden), all x-claude-code-*,
      and internal Brigade headers (x-enhanced-token, x-brigade-run-id).
    - Strips all HOP-BY-HOP headers (RFC 9113 ?8.2.2).
    - Resolves *resolved.api_base* via *resolved.api_base_env* before use.
    - Injects the provider credential from *resolved.api_key_env*.
    - Validates the upstream host matches the pinned endpoint.

    Uses *resolved.upstream_model* as the target model and
    *resolved.api_base* (resolved through env vars) for the endpoint.
    """
    from fastapi.responses import JSONResponse, StreamingResponse
    from urllib.parse import urlsplit

    # ------------------------------------------------------------------
    # 1. Resolve the upstream endpoint — FAIL CLOSED (P0-1)
    #    No Anthropic fallback. If no explicit endpoint is pinned, return 503.
    # ------------------------------------------------------------------
    api_base = resolved.api_base  # already resolved in routing.py via _resolve_api_base
    if not api_base:
        LOGGER.error(
            "proxy_direct_anthropic: no pinned endpoint for model=%s; "
            "refusing to send request without explicit endpoint",
            resolved.model_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "provider_endpoint_unavailable",
                    "message": (
                        "No upstream endpoint is pinned for this direct-anthropic model. "
                        "The Anthropic fallback was removed to prevent credential leakage."
                    ),
                }
            },
        )

    upstream = api_base.rstrip("/")

    # Validate upstream host matches pinned endpoint using urlsplit (P0-1)
    # Compare scheme, hostname, and port — not loosely transformed strings.
    parsed_upstream = urlsplit(upstream)
    parsed_pinned = urlsplit(api_base)
    if (parsed_upstream.hostname != parsed_pinned.hostname
            or parsed_upstream.scheme != parsed_pinned.scheme
            or parsed_upstream.port != parsed_pinned.port):
        LOGGER.error(
            "proxy_direct_anthropic: endpoint mismatch — "
            "pinned=%s://%s:%s resolved=%s://%s:%s; refusing",
            parsed_pinned.scheme, parsed_pinned.hostname, parsed_pinned.port,
            parsed_upstream.scheme, parsed_upstream.hostname, parsed_upstream.port,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "credential_boundary_violation",
                    "message": (
                        f"Upstream endpoint {parsed_upstream.hostname}:{parsed_upstream.port} "
                        f"does not match pinned endpoint {parsed_pinned.hostname}:{parsed_pinned.port}"
                    ),
                }
            },
        )

    # ------------------------------------------------------------------
    # 2. Rewrite model
    # ------------------------------------------------------------------
    path = request.url.path
    query = request.url.query
    url = f"{upstream}{path}" + (f"?{query}" if query else "")
    payload["model"] = resolved.upstream_model or payload.get("model")

    # Normalize BEFORE serialization — apply LongCat-style normalization
    # for longcat upstream models.
    if resolved.upstream_model and "longcat" in resolved.upstream_model.lower():
        payload = _normalize_for_direct(payload)

    # ------------------------------------------------------------------
    # 3. Strip ALL sensitive headers from incoming request
    #    Uses the shared DIRECT_ANTHROPIC sanitization mode.
    # ------------------------------------------------------------------
    raw_headers = dict(request.headers)

    cleaned = sanitize_headers_for_backend(raw_headers, SanitizeMode.DIRECT_ANTHROPIC)

    # ------------------------------------------------------------------
    # 4. Inject provider credential using the PINNED auth spec (P0-2)
    #    The auth specification is pinned in the binding at creation time.
    #    Do NOT infer auth from the backend category.
    # ------------------------------------------------------------------
    try:
        header_name, header_value = _resolve_pinned_auth(
            resolved.auth_spec_json,
            resolved.api_key_env,
            "direct-anthropic",
        )
        cleaned[header_name] = header_value
    except ValueError as exc:
        if "No credential configured" not in str(exc):
            LOGGER.warning(
                "proxy_direct_anthropic: credential resolution failed: %s", exc,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "credential_unavailable",
                        "message": str(exc),
                    }
                },
            )

    # ------------------------------------------------------------------
    # 5. Forward to upstream
    # ------------------------------------------------------------------
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    is_streaming = bool(payload.get("stream"))
    started_at = time.perf_counter()

    client = get_upstream_client()
    try:
        admission_id = await _acquire_provider_request(
            resolved, request, streaming=is_streaming,
        )
    except RuntimeError as exc:
        return JSONResponse(status_code=429, content={"error": {"type": "provider_queue_timeout", "message": str(exc)}})
    try:
        upstream_response = await _send_with_provider_retry(
            client,
            lambda: client.build_request(request.method, url, headers=cleaned, content=body),
            resolved,
            started_at,
            admission_id,
        )
    except Exception as exc:
        await _release_provider_request(resolved, admission_id)
        LOGGER.warning(
            "proxy_direct_anthropic error upstream=%s error=%s",
            upstream, exc,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_connection_error",
                    "message": str(exc),
                }
            },
        )

    response_headers = copy_response_headers(upstream_response.headers)
    if is_streaming:
        return StreamingResponse(
            _stream_upstream_with_admission(
                upstream_response, admission_id, resolved, started_at
            ) if admission_id else _stream_upstream(upstream_response),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await _read_with_deadline(upstream_response, resolved, started_at)
    finally:
        await upstream_response.aclose()
        await _release_provider_request(resolved, admission_id)
    try:
        response_payload = json.loads(content)
    except (TypeError, ValueError):
        response_payload = None
    if isinstance(response_payload, dict):
        usage_route = _route_with_reported_deployment(resolved, upstream_response.headers)
        _record_endpoint_usage(
            usage_route,
            admission_id,
            response_payload,
            started_at,
            succeeded=200 <= upstream_response.status_code < 300,
        )
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def proxy_litellm_messages(
    request: Any,
    payload: dict[str, Any],
    resolved: "ResolvedRoute",
) -> Any:
    """Forward through a LiteLLM proxy instance.

    Replaces the role alias model ID with the LiteLLM model-group name
    (``brigade-{registry_key}``), strips internal Brigade headers, and
    POSTs to the active LiteLLM child's ``/v1/messages`` endpoint.

    Streams the Anthropic-formatted response back to Claude Code.
    """
    from fastapi.responses import JSONResponse, StreamingResponse

    base_url = resolved.litellm_base_url
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "litellm_not_available",
                    "message": (
                        "No LiteLLM deployment is available for this binding. "
                        "The LiteLLM catalog may not have finished loading, or "
                        "the generation has been fully drained."
                    ),
                }
            },
        )

    path = request.url.path
    query = request.url.query
    url = f"{base_url}{path}" + (f"?{query}" if query else "")

    # Replace model with LiteLLM model-group name
    model_name = resolved.litellm_model_name or f"brigade-{resolved.model_id}"
    payload["model"] = model_name

    # Build headers: from the request, sanitized, plus LiteLLM auth
    raw_headers = dict(request.headers)
    litellm_key = os.environ.get("BRIGADE_LITELLM_KEY", "")
    headers = sanitize_headers_for_backend(
        raw_headers,
        SanitizeMode.LITELLM_INTERNAL,
        inject_auth_header="authorization" if litellm_key else None,
        inject_auth_value=f"Bearer {litellm_key}" if litellm_key else None,
    )

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    is_streaming = bool(payload.get("stream"))
    started_at = time.perf_counter()

    client = get_upstream_client()
    try:
        admission_id = await _acquire_provider_request(
            resolved, request, streaming=is_streaming,
        )
    except RuntimeError as exc:
        return JSONResponse(status_code=429, content={"error": {"type": "provider_queue_timeout", "message": str(exc)}})
    try:
        upstream_response = await _send_with_provider_retry(
            client,
            lambda: client.build_request(request.method, url, headers=headers, content=body),
            resolved,
            started_at,
            admission_id,
        )
    except Exception as exc:
        await _release_provider_request(resolved, admission_id)
        LOGGER.error(
            "litellm proxy error model=%s base_url=%s error=%s",
            model_name, base_url, exc,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_connection_error",
                    "message": f"LiteLLM connection failed: {exc}",
                }
            },
        )

    response_headers = copy_response_headers(upstream_response.headers)
    if is_streaming:
        return StreamingResponse(
            _stream_upstream_with_admission(
                upstream_response, admission_id, resolved, started_at
            ) if admission_id else _stream_upstream(upstream_response),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await _read_with_deadline(upstream_response, resolved, started_at)
    finally:
        await upstream_response.aclose()
        await _release_provider_request(resolved, admission_id)
    try:
        response_payload = json.loads(content)
    except (TypeError, ValueError):
        response_payload = None
    if isinstance(response_payload, dict):
        usage_route = _route_with_reported_deployment(resolved, upstream_response.headers)
        _record_endpoint_usage(
            usage_route,
            admission_id,
            response_payload,
            started_at,
            succeeded=200 <= upstream_response.status_code < 300,
        )
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_for_direct(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply LongCat-style normalization to a direct-Anthropic payload."""
    messages = payload.get("messages", [])
    if isinstance(messages, list):
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                msg = {k: v for k, v in msg.items() if k != "cache_control"}
                content = msg.get("content")
                if isinstance(content, list):
                    content = [
                        {
                            k: v
                            for k, v in (
                                c.items() if isinstance(c, dict) else {"text": str(c)}
                            ).items()
                            if k != "cache_control"
                        }
                        for c in content
                    ]
                    msg["content"] = content
            cleaned.append(msg)
        payload["messages"] = cleaned

    tools = payload.get("tools", [])
    if isinstance(tools, list):
        cleaned_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                tool = {
                    k: v
                    for k, v in tool.items()
                    if k not in ("cache_control", "strict", "defer_loading")
                }
            cleaned_tools.append(tool)
        payload["tools"] = cleaned_tools

    return payload
