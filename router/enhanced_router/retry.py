"""Shared retry/executor logic for outbound HTTP requests.

Provides a unified retry policy with:
- Transport error retry with connection pool recreation
- Retry-After header support
- Bounded retry for 429/502/503/529
- Exponential backoff with jitter
- Cancellation propagation
- Per-provider timeout policy
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from enhanced_router.base import HOP_BY_HOP


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.1  # fraction of delay to add as jitter
    retryable_statuses: tuple[int, ...] = (429, 502, 503, 529)
    timeout: httpx.Timeout = httpx.Timeout(
        connect=30.0, read=90.0, write=120.0, pool=30.0
    )


@dataclass
class RetryResult:
    """Result of a retryable operation."""
    response: Optional[httpx.Response] = None
    error: Optional[BaseException] = None
    attempts: int = 0
    total_time: float = 0.0


def parse_retry_after(headers: httpx.Headers) -> Optional[float]:
    """Parse Retry-After header value to seconds."""
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
        now = time.time()
        diff = dt.timestamp() - now
        return diff if diff > 0 else None
    except Exception:
        return None


def compute_delay(attempt: int, policy: RetryPolicy, retry_after: Optional[float] = None) -> float:
    """Compute delay with exponential backoff and jitter."""
    if retry_after is not None:
        delay = retry_after
    else:
        delay = policy.base_delay * (2 ** attempt)
    delay = max(policy.base_delay, min(delay, policy.max_delay))
    # Add jitter
    jitter = delay * policy.jitter * (2 * random.random() - 1)
    return delay + jitter


async def execute_with_retry(
    client: httpx.AsyncClient,
    request_builder: Callable[[], httpx.Request],
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, Exception], Any]] = None,
) -> RetryResult:
    """Execute an HTTP request with retry logic.

    Args:
        client: The httpx AsyncClient to use.
        request_builder: Function that returns a new httpx.Request for each attempt.
        policy: Retry policy configuration.
        on_retry: Optional callback(attempt, exception) called before each retry.

    Returns:
        RetryResult with response or error.
    """
    policy = policy or RetryPolicy()
    start_time = time.monotonic()
    last_error: Optional[BaseException] = None

    # ``max_retries`` means retries after the initial attempt.  Even a
    # development policy of zero therefore still sends one request.
    max_attempts = max(1, policy.max_retries + 1)
    for attempt in range(max_attempts):
        try:
            request = request_builder()
            response = await client.send(request, stream=True)

            # Check for retryable status
            if response.status_code in policy.retryable_statuses and attempt < policy.max_retries:
                retry_after = parse_retry_after(response.headers)
                await response.aclose()

                if on_retry:
                    await on_retry(attempt, Exception(f"HTTP {response.status_code}"))

                delay = compute_delay(attempt, policy, retry_after)
                await asyncio.sleep(delay)
                continue

            # Success or non-retryable status
            total_time = time.monotonic() - start_time
            return RetryResult(response=response, attempts=attempt + 1, total_time=total_time)

        except httpx.HTTPError as exc:
            last_error = exc
            if on_retry:
                await on_retry(attempt, exc)

            if attempt < policy.max_retries:
                delay = compute_delay(attempt, policy)
                await asyncio.sleep(delay)
                continue

            total_time = time.monotonic() - start_time
            return RetryResult(error=exc, attempts=attempt + 1, total_time=total_time)

        except asyncio.CancelledError:
            # Cancellation is a caller decision, not an upstream failure.
            # Never sleep, retry, or convert it into a normal result.
            raise

    total_time = time.monotonic() - start_time
    return RetryResult(error=last_error or Exception("Max retries exceeded"), attempts=max_attempts, total_time=total_time)


async def stream_with_retry(
    response: httpx.Response,
    retry_after: Optional[float] = None,
) -> AsyncIterator[bytes]:
    """Stream response body with proper cleanup."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def sanitize_headers_for_backend(
    headers: dict[str, str],
    mode: str,
    *,
    inject_auth_header: Optional[str] = None,
    inject_auth_value: Optional[str] = None,
) -> dict[str, str]:
    """Sanitize headers based on backend mode.

    Modes:
    - anthropic_passthrough: preserve Anthropic headers, strip Brigade headers
    - external_provider: strip all Claude credentials and internal identity
    - litellm_internal: strip auth, inject internal LiteLLM key
    - direct_anthropic: like external_provider but for LongCat-style providers
    """
    cleaned: dict[str, str] = {}

    # Always strip hop-by-hop headers
    for key, value in headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP:
            continue

        if mode == "anthropic_passthrough":
            # Preserve most Anthropic headers
            if lower.startswith("anthropic-"):
                cleaned[key] = value
                continue
            # Strip Brigade headers
            if lower in {"x-enhanced-token", "x-brigade-run-id", "x-brigade-session-id"}:
                continue
            # Strip x-claude-code-* headers
            if lower.startswith("x-claude-code-"):
                continue
            # Preserve standard headers
            cleaned[key] = value

        elif mode == "external_provider":
            # Strip ALL Claude OAuth headers
            if lower in {"authorization", "x-api-key"}:
                continue
            # Strip x-claude-code-* headers
            if lower.startswith("x-claude-code-"):
                continue
            # Strip Brigade headers
            if lower in {"x-enhanced-token", "x-brigade-run-id", "x-brigade-session-id"}:
                continue
            # Strip unsupported anthropic-* headers except version
            if lower.startswith("anthropic-") and lower != "anthropic-version":
                continue
            cleaned[key] = value

        elif mode == "litellm_internal":
            # Strip incoming authorization
            if lower in {"authorization", "x-api-key"}:
                continue
            # Strip x-claude-code-* headers
            if lower.startswith("x-claude-code-"):
                continue
            # Strip Brigade headers
            if lower in {"x-enhanced-token", "x-brigade-run-id", "x-brigade-session-id"}:
                continue
            # Strip all anthropic-* headers
            if lower.startswith("anthropic-"):
                continue
            cleaned[key] = value

        elif mode == "direct_anthropic":
            # Same as external_provider
            if lower in {"authorization", "x-api-key"}:
                continue
            if lower.startswith("x-claude-code-"):
                continue
            if lower in {"x-enhanced-token", "x-brigade-run-id", "x-brigade-session-id"}:
                continue
            if lower.startswith("anthropic-") and lower != "anthropic-version":
                continue
            cleaned[key] = value

        else:
            raise ValueError(f"Unknown sanitize mode: {mode}")

    # Inject content-type for non-passthrough modes
    if mode != "anthropic_passthrough":
        cleaned.setdefault("content-type", "application/json")

    # Inject auth credentials if provided
    if inject_auth_header and inject_auth_value:
        cleaned[inject_auth_header] = inject_auth_value

    return cleaned


class OutboundExecutor:
    """High-level executor for outbound HTTP requests with retry and health tracking."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        policy: Optional[RetryPolicy] = None,
        provider_name: str = "unknown",
    ):
        self.client = client
        self.policy = policy or RetryPolicy()
        self.provider_name = provider_name
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
        }

    async def execute(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        stream: bool = False,
        sanitize_mode: str = "external_provider",
        inject_auth_header: Optional[str] = None,
        inject_auth_value: Optional[str] = None,
    ) -> httpx.Response:
        """Execute a request with full retry logic."""
        self._metrics["total_requests"] += 1

        sanitized_headers = sanitize_headers_for_backend(
            headers,
            sanitize_mode,
            inject_auth_header=inject_auth_header,
            inject_auth_value=inject_auth_value,
        )

        def build_request() -> httpx.Request:
            return self.client.build_request(method, url, headers=sanitized_headers, content=body)

        result = await execute_with_retry(
            self.client,
            build_request,
            self.policy,
            on_retry=lambda attempt, exc: self._record_retry(),
        )

        if result.error:
            self._metrics["failed_requests"] += 1
            if isinstance(result.error, asyncio.CancelledError):
                raise
            raise httpx.HTTPError(str(result.error)) from result.error

        self._metrics["successful_requests"] += 1
        self._metrics["total_retries"] += result.attempts - 1

        if result.response is None:
            raise httpx.HTTPError("No response received after retries")
        return result.response

    def _record_retry(self):
        self._metrics["total_retries"] += 1

    def get_metrics(self) -> dict:
        return self._metrics.copy()
