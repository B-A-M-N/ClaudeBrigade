"""Explicit, credential-owned provider catalog discovery.

Discovery is never performed implicitly by a health probe. The caller must
request it, and the returned catalog is persisted as evidence for a registry
generation. Provider metadata is treated as authoritative for context limits;
bundled YAML does not invent those values.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class DiscoveredEndpoint:
    provider_id: str
    model_id: str
    endpoint_id: str
    display_name: str | None
    context_tokens: int | None
    max_output_tokens: int | None
    capabilities: dict[str, Any]
    raw: dict[str, Any]
    logical_model_id: str | None = None


def _first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def normalize_model(provider_id: str, endpoint_id: str, payload: dict[str, Any]) -> DiscoveredEndpoint | None:
    model_id = payload.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    context = _first_int(payload, "context_length", "context_window", "max_context_tokens", "context_tokens")
    output = _first_int(payload, "max_output_tokens", "max_completion_tokens", "output_tokens")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    return DiscoveredEndpoint(
        provider_id=provider_id,
        model_id=model_id.strip(),
        endpoint_id=endpoint_id,
        display_name=payload.get("name") if isinstance(payload.get("name"), str) else None,
        context_tokens=context,
        max_output_tokens=output,
        capabilities=capabilities,
        raw=payload,
    )


def discover_openai_models(
    *,
    provider_id: str,
    endpoint_id: str,
    catalog_url: str,
    api_key: str,
    timeout: float = 30.0,
    max_attempts: int = 1,
    retryable_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504, 529),
    max_backoff_seconds: float = 5.0,
) -> list[DiscoveredEndpoint]:
    """Fetch an OpenAI-compatible catalog with bounded transient retries.

    Explicit catalog URLs are used exactly as configured.  Only transport
    failures and the configured transient HTTP statuses advance the retry
    loop; authentication and endpoint/path errors remain immediate failures so
    the CLI can show the real problem instead of masking it with retries.
    """
    attempts = max(1, int(max_attempts))
    retryable = set(retryable_statuses)
    response: httpx.Response | None = None
    for attempt in range(attempts):
        try:
            response = httpx.get(
                catalog_url,
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                timeout=timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in retryable or attempt + 1 >= attempts:
                raise
            retry_after = exc.response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else None
            except ValueError:
                delay = None
            ceiling = min(max_backoff_seconds, 0.5 * (2**attempt))
            delay = max(0.1, min(delay if delay is not None else ceiling, max_backoff_seconds))
            time.sleep(random.uniform(0.0, delay))
        except httpx.RequestError:
            if attempt + 1 >= attempts:
                raise
            ceiling = min(max_backoff_seconds, 0.5 * (2**attempt))
            time.sleep(random.uniform(0.0, max(0.1, ceiling)))
    if response is None:
        raise RuntimeError("provider catalog request produced no response")
    payload = response.json()
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("provider /models response did not contain a data array")
    result = [normalize_model(provider_id, endpoint_id, item) for item in entries if isinstance(item, dict)]
    return [entry for entry in result if entry is not None]


def response_digest(entries: list[DiscoveredEndpoint]) -> str:
    raw = [entry.raw for entry in sorted(entries, key=lambda item: item.model_id)]
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def persist_catalog(state: Any, entries: list[DiscoveredEndpoint]) -> str:
    digest = response_digest(entries)
    state.store_provider_catalog_entries(
        provider_id=entries[0].provider_id if entries else "",
        response_digest=digest,
        entries=entries,
    )
    return digest
