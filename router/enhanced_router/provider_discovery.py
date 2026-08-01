"""Explicit, credential-owned provider catalog discovery.

Discovery is never performed implicitly by a health probe. The caller must
request it, and the returned catalog is persisted as evidence for a registry
generation. Provider metadata is treated as authoritative for context limits;
bundled YAML does not invent those values.
"""

from __future__ import annotations

import hashlib
import json
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
    *, provider_id: str, endpoint_id: str, base_url: str, api_key: str, timeout: float = 30.0
) -> list[DiscoveredEndpoint]:
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=timeout,
        follow_redirects=False,
    )
    response.raise_for_status()
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
