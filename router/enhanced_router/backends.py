"""Backend dispatch interfaces and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BackendType(Enum):
    """Supported backend kinds."""

    ANTHROPIC_PASSTHROUGH = "anthropic_passthrough"
    DIRECT_ANTHROPIC = "direct_anthropic"
    LITELLM = "litellm"


@dataclass(frozen=True)
class ResolvedRoute:
    """Resolved routing decision for an incoming request."""

    kind: BackendType
    role: str | None = None
    model_id: str | None = None
    upstream_model: str | None = None
    api_base: str | None = None
    agent_binding_id: int | None = None
    route_version: int | None = None
    registry_hash: str | None = None
    catalog_generation: int | None = None
    litellm_model_name: str | None = None
    litellm_base_url: str | None = None


# Maps the public-facing model aliases to internal role strings.
ROLE_MODEL_ALIASES: dict[str, str] = {
    "anthropic-brigade-recon": "recon",
    "anthropic-brigade-implementer": "implementer",
    "anthropic-brigade-adversary": "adversary",
    "anthropic-brigade-repairer": "repairer",
}

# Headers that MUST be stripped before forwarding upstream.
_STRIPTED_HEADERS = frozenset({
    "x-enhanced-token",
    "x-brigade-run-id",
})

# Headers that MUST be preserved even though they are brigade-related.
_ALLOWED_HEADERS = frozenset({
    "anthropic-version",
    "anthropic-beta",
    "content-type",
    "accept",
})


def sanitize_upstream_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove internal Brigade headers while preserving standard ones.

    Parameters
    ----------
    headers :
        Raw request headers as a dict.

    Returns
    -------
    dict[str, str]
        A filtered copy suitable for upstream forwarding.
    """
    cleaned: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _STRIPTED_HEADERS:
            continue
        # Preserve allowed brigade-related headers
        if lower in _ALLOWED_HEADERS:
            cleaned[key] = value
            continue
        cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Stub proxy signatures (fleshed out in later phases)
# ---------------------------------------------------------------------------


async def proxy_anthropic_passthrough(
    request: Any, payload: dict[str, Any]
) -> Any:
    """Forward the request directly to Anthropic upstream."""
    raise NotImplementedError


async def proxy_direct_anthropic(
    request: Any,
    payload: dict[str, Any],
    resolved: "ResolvedRoute",
) -> Any:
    """Route worker traffic to a direct Anthropic-compatible backend (e.g. LongCat).

    Uses *resolved.upstream_model* as the target model and
    *resolved.api_base* or env vars for the endpoint.
    """
    raise NotImplementedError


async def proxy_litellm_messages(
    request: Any,
    payload: dict[str, Any],
    resolved: "ResolvedRoute",
) -> Any:
    """Forward through a LiteLLM proxy instance."""
    raise NotImplementedError
